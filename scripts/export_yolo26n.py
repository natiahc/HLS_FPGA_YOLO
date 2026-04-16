from ultralytics import YOLO


def main():
    # Load pretrained YOLO26n model
    model = YOLO("yolo26n.pt")

    # Export to ONNX
    model.export(
        format="onnx",
        imgsz=640,
        opset=13,
        simplify=True
    )

    print("Export complete: yolo26n.onnx")


if __name__ == "__main__":
    main()
