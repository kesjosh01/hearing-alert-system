
import numpy as np
from scipy.io import wavfile
from pathlib import Path


# ============================================================
# Configuration
# compare_dtw.py와 동일하게 유지
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

HOP_SECONDS = 0.2
HOP_SAMPLES = int(SAMPLE_RATE * HOP_SECONDS)

FRAME_SAMPLES = 160
FRAME_COUNT = 100

FEATURE_DIM = 32
POOL_SIZE = 5

DTW_RADIUS = 5
MAX_REFERENCES = 100

SILENCE_THRESHOLD = int(32768 * 0.005)

FIRE_FILE = Path("recordings/fire_alarm.wav")
DOORBELL_FILE = Path("recordings/doorbell.wav")

OUTPUT = Path("dtw_reference.h")


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

    return audio.astype(np.float32)


# ============================================================
# Remove silence
# ============================================================

def trim_silence(audio):

    frame_count = len(audio) // FRAME_SAMPLES

    if frame_count == 0:
        return np.array([], dtype=np.float32)

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
        raise ValueError("No sound detected!")

    start = active_indices[0] * FRAME_SAMPLES
    end = (active_indices[-1] + 1) * FRAME_SAMPLES

    return audio[start:end]


# ============================================================
# Feature extraction
# Exactly matches compare_dtw.py
# ============================================================

def make_features(audio):

    if len(audio) < WINDOW_SAMPLES:
        return None

    audio = audio[:WINDOW_SAMPLES].astype(np.float32)

    audio = audio - np.mean(audio)

    peak = np.max(np.abs(audio))

    if peak < 1:
        return np.zeros(
            (FRAME_COUNT, FEATURE_DIM),
            dtype=np.int8
        )

    audio = audio / peak

    frames = audio.reshape(
        FRAME_COUNT,
        FRAME_SAMPLES
    )

    pooled = frames.reshape(
        FRAME_COUNT,
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
# Generate reference bank
# ============================================================

def generate_references(filename):

    audio = trim_silence(
        load_wav(filename)
    )

    references = []

    for start in range(
        0,
        len(audio) - WINDOW_SAMPLES + 1,
        HOP_SAMPLES
    ):

        segment = audio[
            start:start + WINDOW_SAMPLES
        ]

        features = make_features(segment)

        if features is not None:
            references.append(features)

        if len(references) >= MAX_REFERENCES:
            break

    if not references:
        raise ValueError(
            f"No references generated: {filename}"
        )

    bank = np.array(
        references,
        dtype=np.int8
    )

    print(f"\n{filename}")
    print(f"Trimmed duration: {len(audio) / SAMPLE_RATE:.2f} sec")
    print(f"Reference count: {len(bank)}")
    print(f"Feature shape: {bank.shape}")
    print(f"Feature dtype: {bank.dtype}")

    return bank


# ============================================================
# Convert reference bank to C array
# ============================================================

def format_reference_array(name, bank):

    count = len(bank)

    lines = [
        f"static const int8_t {name}"
        f"[{count}][DTW_FRAME_COUNT][DTW_FEATURE_DIM] = {{"
    ]

    for template in bank:

        lines.append("    {")

        for frame in template:

            values = ", ".join(
                str(int(value)) for value in frame
            )

            lines.append(f"        {{{values}}},")

        lines.append("    },")

    lines.append("};")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():

    print("Generating DTW references...")

    fire_bank = generate_references(FIRE_FILE)
    doorbell_bank = generate_references(DOORBELL_FILE)

    print("\n========== REFERENCE SUMMARY ==========")

    print(f"FIRE references: {len(fire_bank)}")
    print(f"DOORBELL references: {len(doorbell_bank)}")

    print(f"Feature shape: {fire_bank[0].shape}")
    print(f"Feature dtype: {fire_bank.dtype}")
    print(f"DTW radius: {DTW_RADIUS}")

    header = [
        "#ifndef DTW_REFERENCE_H",
        "#define DTW_REFERENCE_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define DTW_FRAME_COUNT {FRAME_COUNT}",
        f"#define DTW_FEATURE_DIM {FEATURE_DIM}",
        f"#define DTW_RADIUS {DTW_RADIUS}",
        f"#define FIRE_REF_COUNT {len(fire_bank)}",
        f"#define DOORBELL_REF_COUNT {len(doorbell_bank)}",
        "",
        format_reference_array("fire_refs", fire_bank),
        "",
        format_reference_array("doorbell_refs", doorbell_bank),
        "",
        "#endif",
        ""
    ]

    OUTPUT.write_text(
        "\n".join(header),
        encoding="utf-8"
    )

    print("\nReference generation complete!")
    print(f"Output: {OUTPUT.resolve()}")
    print(f"Header size: {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
