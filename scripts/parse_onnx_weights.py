import onnx
import numpy as np
from onnx import numpy_helper


def main():
    model = onnx.load("yolo26n.onnx")

    params = {}

    print("Reading ONNX initializers...\n")
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        params[init.name] = arr
        print(f"{init.name:60s} shape={arr.shape}")

    np.savez("yolo26n_params.npz", **params)
    print("\nSaved parameters to yolo26n_params.npz")


if __name__ == "__main__":
    main()
