
import numpy as np
from pathlib import Path

FILES = {
    "fire_alarm_reference.npy": "fire_alarm_reference.h",
    "doorbell_reference.npy": "doorbell_reference.h",
}

for input_file, output_file in FILES.items():
    data = np.load(input_file)

    if data.shape != (64, 256):
        raise ValueError(f"Unexpected shape: {input_file}")

    if data.dtype != np.uint8:
        raise ValueError(f"Unexpected data type: {input_file}")

    # 2차원 배열을 1차원 C 배열로 변환
    values = data.flatten()

    variable_name = Path(output_file).stem

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"#define {variable_name.upper()}_WIDTH 256\n")
        f.write(f"#define {variable_name.upper()}_HEIGHT 64\n")
        f.write(f"#define {variable_name.upper()}_SIZE 16384\n\n")

        f.write(
            f"static const uint8_t {variable_name}[16384] = {{\n"
        )

        for i in range(0, len(values), 16):
            line = ", ".join(str(int(v)) for v in values[i:i + 16])
            f.write("    " + line + ",\n")

        f.write("};\n")

    print(f"Created: {output_file}")

print("\nAll header files generated successfully!")
