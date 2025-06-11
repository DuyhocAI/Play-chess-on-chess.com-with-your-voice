from ultralytics import YOLO
import torch

def main():
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("Không có GPU, đang dùng CPU")

    model = YOLO("yolov8n.pt")  

    data_yaml_path = r"C:\BaoDuy\Autochess\2D Chessboard and Chess Pieces.v4i.yolov8\data.yaml"

    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0  
    )

    model.predict(
        source=r"C:\BaoDuy\Autochess\2D Chessboard and Chess Pieces.v4i.yolov8\test\images",
        conf=0.25,
        save=True
    )

    print("Done!")

if __name__ == '__main__':
    main()
