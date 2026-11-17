
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.2

FRAME_SECONDS = 0.01
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_SECONDS)

FEATURE_DIM = 32
POOL_SIZE = 5

DTW_RADIUS = 5
MAX_REFERENCES = 100

SILENCE_THRESHOLD = int(32768 * 0.005)

# Reference recordings
FIRE_FILE = "recordings/fire_alarm.wav"
DOORBELL_FILE = "recordings/doorbell.wav"

# Independent test recordings
FIRE_TEST_FILE = "recordings/fire_alarm_test.wav"
DOORBELL_TEST_FILE = "recordings/doorbell_test.wav"


# ============================================================
# Load WAV
# ============================================================

def load_wav(filename):

    sample_rate, audio = wavfile.read(filename)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{filename}: Expected {SAMPLE_RATE} Hz, "
            f"but got {sample_rate} Hz"
        )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    print(f"\n{filename}")
    print(f"Sample rate: {sample_rate}")
    print(f"Duration: {len(audio) / sample_rate:.2f} sec")

    return audio


# ============================================================
# Remove leading and trailing silence
# ============================================================

def trim_silence(audio):

    frame_count = len(audio) // FRAME_SAMPLES

    if frame_count == 0:
        return audio

    usable = audio[:frame_count * FRAME_SAMPLES]

    frames = usable.reshape(
        frame_count,
        FRAME_SAMPLES
    )

    active = (
        np.max(np.abs(frames), axis=1)
        > SILENCE_THRESHOLD
    )

    active_indices = np.where(active)[0]

    if len(active_indices) == 0:
        return np.array([], dtype=np.float32)

    start = active_indices[0] * FRAME_SAMPLES
    end = (active_indices[-1] + 1) * FRAME_SAMPLES

    trimmed = audio[start:end]

    print(
        f"Trimmed duration: "
        f"{len(trimmed) / SAMPLE_RATE:.2f} sec"
    )

    return trimmed


# ============================================================
# Feature extraction
# 160 samples -> 32 averaged features
# ============================================================

def make_features(audio):

    window_samples = int(
        SAMPLE_RATE * WINDOW_SECONDS
    )

    if len(audio) < window_samples:
        return None

    audio = audio[:window_samples].astype(np.float32)

    audio = audio - np.mean(audio)

    peak = np.max(np.abs(audio))

    if peak < 1:
        return np.zeros(
            (100, FEATURE_DIM),
            dtype=np.int8
        )

    audio = audio / peak

    frames = audio.reshape(
        100,
        FRAME_SAMPLES
    )

    pooled = frames.reshape(
        100,
        FEATURE_DIM,
        POOL_SIZE
    ).mean(axis=2)

    features = np.clip(
        np.round(pooled * 127),
        -127,
        127
    ).astype(np.int8)

    return features


# ============================================================
# Generate reference templates
# ============================================================

def generate_references(audio):

    window_samples = int(
        SAMPLE_RATE * WINDOW_SECONDS
    )

    hop_samples = int(
        SAMPLE_RATE * HOP_SECONDS
    )

    references = []

    for start in range(
        0,
        len(audio) - window_samples + 1,
        hop_samples
    ):

        segment = audio[
            start:start + window_samples
        ]

        features = make_features(segment)

        if features is not None:
            references.append(features)

        if len(references) >= MAX_REFERENCES:
            break

    return references


# ============================================================
# DTW distance
# ============================================================

def dtw_distance(reference, test):

    n = len(reference)
    m = len(test)

    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0

    for i in range(1, n + 1):

        current = np.full(m + 1, np.inf)

        start_j = max(1, i - DTW_RADIUS)
        end_j = min(m, i + DTW_RADIUS)

        for j in range(start_j, end_j + 1):

            difference = (
                reference[i - 1].astype(np.int16)
                - test[j - 1].astype(np.int16)
            )

            local_cost = np.mean(
                np.abs(difference)
            )

            current[j] = local_cost + min(
                previous[j],
                current[j - 1],
                previous[j - 1]
            )

        previous = current

    return previous[m] / max(n, m)


# ============================================================
# Classification
# ============================================================

def classify(test_features, fire_refs, doorbell_refs):

    fire_distance = min(
        dtw_distance(ref, test_features)
        for ref in fire_refs
    )

    doorbell_distance = min(
        dtw_distance(ref, test_features)
        for ref in doorbell_refs
    )

    if fire_distance < doorbell_distance:
        result = "FIRE ALARM"
    else:
        result = "DOORBELL"

    return fire_distance, doorbell_distance, result


# ============================================================
# Evaluate independent test recording
# ============================================================

def evaluate_recording(
    filename,
    expected_class,
    fire_refs,
    doorbell_refs
):

    audio = load_wav(filename)
    audio = trim_silence(audio)

    window_samples = int(
        SAMPLE_RATE * WINDOW_SECONDS
    )

    hop_samples = int(
        SAMPLE_RATE * HOP_SECONDS
    )

    print(f"\n{expected_class} TEST")
    print("=" * 75)

    correct = 0
    total = 0

    results = []

    for start in range(
        0,
        len(audio) - window_samples + 1,
        hop_samples
    ):

        segment = audio[
            start:start + window_samples
        ]

        features = make_features(segment)

        if features is None:
            continue

        fire_dist, doorbell_dist, result = classify(
            features,
            fire_refs,
            doorbell_refs
        )

        if result == expected_class:
            correct += 1

        total += 1

        time_position = start / SAMPLE_RATE

        print(
            f"{time_position:5.2f}s | "
            f"FIRE: {fire_dist:8.4f} | "
            f"DOORBELL: {doorbell_dist:8.4f} | "
            f"Result: {result}"
        )

        results.append(
            (
                time_position,
                fire_dist,
                doorbell_dist,
                result
            )
        )

    accuracy = (
        correct / total * 100
        if total > 0 else 0
    )

    print("-" * 75)
    print(
        f"Correct windows: {correct}/{total} "
        f"({accuracy:.1f}%)"
    )

    return results, correct, total


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load reference recordings only
    # --------------------------------------------------------

    print("\n========== REFERENCE DATA ==========")

    fire_audio = trim_silence(
        load_wav(FIRE_FILE)
    )

    doorbell_audio = trim_silence(
        load_wav(DOORBELL_FILE)
    )

    # --------------------------------------------------------
    # 2. Generate templates
    # --------------------------------------------------------

    fire_refs = generate_references(fire_audio)

    doorbell_refs = generate_references(
        doorbell_audio
    )

    print("\nREFERENCE INFORMATION")
    print("=" * 45)

    print(f"FIRE references: {len(fire_refs)}")
    print(f"DOORBELL references: {len(doorbell_refs)}")

    if not fire_refs or not doorbell_refs:
        raise ValueError(
            "Reference templates could not be generated."
        )

    print(f"Feature shape: {fire_refs[0].shape}")
    print(f"Feature dtype: {fire_refs[0].dtype}")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"DTW radius: {DTW_RADIUS}")

    # --------------------------------------------------------
    # 3. Evaluate independent test recordings
    # --------------------------------------------------------

    print("\n========== INDEPENDENT TEST DATA ==========")

    fire_results, fire_correct, fire_total = (
        evaluate_recording(
            FIRE_TEST_FILE,
            "FIRE ALARM",
            fire_refs,
            doorbell_refs
        )
    )

    doorbell_results, doorbell_correct, doorbell_total = (
        evaluate_recording(
            DOORBELL_TEST_FILE,
            "DOORBELL",
            fire_refs,
            doorbell_refs
        )
    )

    # --------------------------------------------------------
    # 4. Overall accuracy
    # --------------------------------------------------------

    total_correct = fire_correct + doorbell_correct
    total_windows = fire_total + doorbell_total

    overall_accuracy = (
        total_correct / total_windows * 100
        if total_windows > 0 else 0
    )

    print("\n========== FINAL RESULTS ==========")

    print(
        f"FIRE ALARM: "
        f"{fire_correct}/{fire_total}"
    )

    print(
        f"DOORBELL: "
        f"{doorbell_correct}/{doorbell_total}"
    )

    print(
        f"OVERALL: "
        f"{total_correct}/{total_windows} "
        f"({overall_accuracy:.1f}%)"
    )

    # --------------------------------------------------------
    # 5. Plot test distances
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9)
    )

    for ax, results, title in [
        (axes[0], fire_results, "FIRE ALARM TEST"),
        (axes[1], doorbell_results, "DOORBELL TEST")
    ]:

        if results:

            times = [r[0] for r in results]
            fire_distances = [r[1] for r in results]
            doorbell_distances = [r[2] for r in results]

            ax.plot(
                times,
                fire_distances,
                marker="o",
                label="FIRE distance"
            )

            ax.plot(
                times,
                doorbell_distances,
                marker="o",
                label="DOORBELL distance"
            )

        ax.set_title(title)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("DTW Distance")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    plt.savefig(
        "dtw_independent_test_32.png"
    )

    plt.show()

    print(
        "\nSaved: dtw_independent_test_32.png"
    )


if __name__ == "__main__":
    main()
