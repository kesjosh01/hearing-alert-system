
import numpy as np

files = [
    "fire_alarm_reference.npy",
    "doorbell_reference.npy"
]

for filename in files:
    data = np.load(filename)

    print(f"\nFile: {filename}")
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("Minimum:", data.min())
    print("Maximum:", data.max())
    print("File size:", data.nbytes, "bytes")
