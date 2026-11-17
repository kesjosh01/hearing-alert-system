
import serial
import time
import wave
from datetime import datetime
from pathlib import Path

PORT = "COM8"
BAUDRATE = 115200

MARKER = b"S3WAV001"
WAV_HEADER_SIZE = 44
PCM_DATA_SIZE = 16000 * 10 * 2

# 녹음 파일 저장 폴더
RECORD_DIR = Path("recordings")
RECORD_DIR.mkdir(exist_ok=True)

# 기존 파일과 이름이 겹치지 않는 파일명 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_FILE = RECORD_DIR / f"recording_{timestamp}.wav"

counter = 1
while OUTPUT_FILE.exists():
    OUTPUT_FILE = RECORD_DIR / f"recording_{timestamp}_{counter}.wav"
    counter += 1

def read_exact(ser, size):
    data = bytearray()

    while len(data) < size:
        chunk = ser.read(size - len(data))

        if not chunk:
            raise TimeoutError(
                f"Data receive timeout: {len(data)}/{size} bytes"
            )

        data.extend(chunk)

    return bytes(data)


def main():
    print(f"Opening {PORT}...")

    with serial.Serial(
        PORT,
        BAUDRATE,
        timeout=30,
        write_timeout=5
    ) as ser:

        time.sleep(2)
        ser.reset_input_buffer()

        print("Starting recording...")
        ser.write(b"R")
        ser.flush()
        print("Waiting for WAV marker...")

        buffer = bytearray()

        while True:
            byte = ser.read(1)

            if not byte:
                raise TimeoutError("WAV marker not received")

            buffer.extend(byte)

            if len(buffer) > len(MARKER):
                del buffer[0]

            if bytes(buffer) == MARKER:
                break

        print("WAV marker received!")

        print("Receiving WAV header...")
        header = read_exact(ser, WAV_HEADER_SIZE)

        print("Receiving PCM data...")
        pcm_data = read_exact(ser, PCM_DATA_SIZE)

        with open(OUTPUT_FILE, "xb") as f:
            f.write(header)
            f.write(pcm_data)

        print("\nTransfer complete!")
        print(f"Saved: {OUTPUT_FILE}")
        print(f"PCM size: {len(pcm_data)} bytes")


if __name__ == "__main__":
    main()
