from __future__ import annotations

from typing import Any

import cv2

from ultralytics import YOLO

_MODEL: YOLO | None = None


def _ensure_model() -> YOLO:
    if _MODEL is None:
        raise RuntimeError("YOLOv8 模型未初始化，请先在 main() 中加载模型。")
    return _MODEL


def detect_frame(frame):
    model = _ensure_model()
    return model.predict(source=frame, verbose=False)[0]


def _extract_detections(results) -> list[dict[str, Any]]:
    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    names_map = getattr(results, "names", None) or getattr(_ensure_model(), "names", None) or {}

    detections: list[dict[str, Any]] = []
    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
        x1i, y1i, x2i, y2i = (round(v) for v in (x1, y1, x2, y2))
        x_center = round((x1i + x2i) / 2)
        y_center = round((y1i + y2i) / 2)

        if isinstance(names_map, dict):
            name = names_map.get(int(cls_id), str(int(cls_id)))
        else:
            name = str(names_map[int(cls_id)])

        detections.append(
            {
                "name": name,
                "confidence": float(conf),
                "bbox": (x1i, y1i, x2i, y2i),
                "center": (x_center, y_center),
            }
        )

    return detections


def draw_results(frame, results):
    annotated = frame.copy()
    detections = _extract_detections(results)

    h, w = annotated.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{det['name']} {det['confidence']:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cx, cy = det["center"]
        cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

    return annotated


def main() -> None:
    global _MODEL

    # 加载 Ultralytics YOLOv8 模型（会在本地缺失时自动下载权重）
    _MODEL = YOLO("yolov8n.pt")

    # 打开摄像头（Windows 优先使用 DirectShow）
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("摄像头打开失败，请检查设备占用/权限/编号。")

    window_name = "YOLOv8 Real-time Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("❌ 读取摄像头帧失败")
                break

            results = detect_frame(frame)
            detections = _extract_detections(results)

            # 终端输出结构化结果（每个检测目标一行）
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["center"]
                print(
                    f"检测到: {det['name']} | 置信度: {det['confidence']:.2f} | "
                    f"边界框: ({x1}, {y1}, {x2}, {y2}) | 中心: ({cx}, {cy})"
                )

            annotated = draw_results(frame, results)
            cv2.imshow(window_name, annotated)

            # 按 q 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 运行失败: {e}")


def get_detected_objects(results):
    detections = _extract_detections(results)
    return [
        {"name": d["name"], "confidence": d["confidence"], "x": d["center"][0], "y": d["center"][1]} for d in detections
    ]
