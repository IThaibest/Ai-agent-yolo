from ultralytics import YOLO

if __name__ == "__main__":
    # 加载训练好的模型，改为自己的路径
    model = YOLO("yolov8n.pt")
    # 修改为自己的图像或者文件夹的路径
    source = "ultralytics/assets"
    # 运行推理，并附加参数
    model.predict(source, save=True)
