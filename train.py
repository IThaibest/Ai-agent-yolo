from ultralytics import YOLO

if __name__ == '__main__':
    # 数据集配置文件
    data_path = 'ultralytics/cfg/datasets/coco128.yaml'
    # 网络结构配置文件
    model_path = 'ultralytics/cfg/models/v8/yolov8.yaml'

    # 训练模型
    model = YOLO(model_path)  # 加载网络结构配置文件
    model.load('yolov8n.pt')  # 加载预训练模型
    results = model.train(
        data=data_path,  # 加载数据集配置文件
        epochs=100,     # 训练100轮
        imgsz=640,      # 图像输入大小
        device='',      # 使用训练设备（不填自动识别，优先GPU，指定cpu填'cpu'）
        workers=0,      # 线程0
        batch=16,       # 批处理大小
    )