from __future__ import annotations

import cv2
from ultralytics import YOLO

# 模型初始化放在函数外部，避免重复加载
MODEL = YOLO("yolov8n.pt")


def detect_object(target_name: str) -> dict:
    """检测摄像头中的指定物体（只检测一帧）并返回结构化结果。"""
    cap = None
    try:
        target = (target_name or "").strip().lower()
        if not target:
            return {"found": False}

        # 打开摄像头（Windows 优先使用 DirectShow）
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("摄像头打开失败，请检查设备占用/权限/编号。")

        ok, frame = cap.read()
        if not ok:
            return {"found": False}

        results = MODEL.predict(source=frame, verbose=False)[0]
        boxes = getattr(results, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {"found": False}

        names_map = getattr(results, "names", None) or getattr(MODEL, "names", None) or {}

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        best = None
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            if isinstance(names_map, dict):
                name = names_map.get(int(cls_id), str(int(cls_id)))
            else:
                name = str(names_map[int(cls_id)])

            if name.lower() != target:
                continue

            x_center = int(round((float(x1) + float(x2)) / 2))
            y_center = int(round((float(y1) + float(y2)) / 2))

            cand = {
                "found": True,
                "name": name,
                "confidence": float(conf),
                "x": x_center,
                "y": y_center,
            }
            if best is None or cand["confidence"] > best["confidence"]:
                best = cand

        return best if best is not None else {"found": False}
    except Exception:
        return {"found": False}
    finally:
        if cap is not None:
            cap.release()


def main() -> None:
    # 简单测试：按需替换目标类别名称，例如 person / car / bottle
    result = detect_object("person")
    print(result)


if __name__ == "__main__":
    main()
