
import wave
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIRE_ALARM = Path("recordings/fire_alarm.wav")
DOORBELL = Path("recordings/doorbell.wav")

WIDTH = 256
HEIGHT = 64

# 무음 판정 기준: 16-bit PCM 최대 진폭의 0.5%
SILENCE_THRESHOLD = 32768 * 0.005


def load_wav(path):
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if rate != 16000 or channels != 1 or width != 2:
        raise ValueError(f"Invalid WAV format: {path}")

    return np.frombuffer(frames, dtype="<i2").astype(np.float32)


def remove_silence(samples):
    # 10ms 단위로 구간별 최대 진폭 계산
    frame_size = 160

    energies = []

    for i in range(0, len(samples), frame_size):
        frame = samples[i:i + frame_size]

        if len(frame) == 0:
            continue

        energies.append(np.max(np.abs(frame)))

    energies = np.array(energies)

    active = np.where(energies > SILENCE_THRESHOLD)[0]

    if len(active) == 0:
        raise ValueError("No sound detected!")

    start = active[0] * frame_size
    end = min((active[-1] + 1) * frame_size, len(samples))

    return samples[start:end]


def make_waveform_image(samples):
    # 진폭 정규화
    peak = np.max(np.abs(samples))

    if peak == 0:
        raise ValueError("Silent audio!")

    normalized = samples / peak

    # 시간축을 256개 구간으로 변환
    chunks = np.array_split(normalized, WIDTH)

    image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for x, chunk in enumerate(chunks):
        amplitude = np.max(np.abs(chunk))

        y = int(amplitude * (HEIGHT // 2 - 1))

        center = HEIGHT // 2

        top = max(0, center - y)
        bottom = min(HEIGHT - 1, center + y)

        image[top:bottom + 1, x] = 255

    return image


def process(path):
    samples = load_wav(path)

    original_duration = len(samples) / 16000

    trimmed = remove_silence(samples)

    trimmed_duration = len(trimmed) / 16000

    image = make_waveform_image(trimmed)

    print(f"\n{path.name}")
    print(f"Original duration: {original_duration:.2f}s")
    print(f"Sound duration: {trimmed_duration:.2f}s")

    return image


def main():
    fire_image = process(FIRE_ALARM)
    doorbell_image = process(DOORBELL)

    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.imshow(fire_image, cmap="gray", aspect="auto")
    plt.title("Fire Alarm - Normalized Waveform")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.imshow(doorbell_image, cmap="gray", aspect="auto")
    plt.title("Doorbell - Normalized Waveform")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("waveform_comparison.png", dpi=200)
    plt.show()

    np.save("fire_alarm_reference.npy", fire_image)
    np.save("doorbell_reference.npy", doorbell_image)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
