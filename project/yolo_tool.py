from __future__ import annotations

from pathlib import Path

import cv2

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "yolov8n.pt"
MODEL_SOURCE = str(MODEL_FILE) if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 1_000_000 else "yolov8n.pt"
MODEL = YOLO(MODEL_SOURCE)


def detect_object(target_name: str) -> dict:
    cap = None
    try:
        target = (target_name or "").strip().lower()
        if not target:
            return {"found": False}
        alias = {"phone": "cell phone"}
        target = alias.get(target, target)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"found": False}

        ok, frame = cap.read()
        if not ok:
            return {"found": False}

        result = MODEL.predict(source=frame, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {"found": False}

        names_map = getattr(result, "names", None) or getattr(MODEL, "names", None) or {}
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        best = None
        thr = 0.6
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            name = (
                str(names_map.get(int(cls_id), cls_id)) if isinstance(names_map, dict) else str(names_map[int(cls_id)])
            )
            if name.lower() != target:
                continue
            if float(conf) < thr:
                continue
            x = round((float(x1) + float(x2)) / 2)
            y = round((float(y1) + float(y2)) / 2)
            cand = {"found": True, "name": name, "confidence": round(float(conf), 4), "x": x, "y": y}
            if best is None or cand["confidence"] > best["confidence"]:
                best = cand

        return best if best else {"found": False}
    finally:
        if cap is not None:
            cap.release()
