"""
كشف اللوحة بنموذج YOLOv8 المصدّر إلى ONNX.
خفيف: يحتاج onnxruntime فقط (بدون PyTorch)، مناسب لـ Streamlit Cloud.
ضع الملف best.onnx بجانب app.py.
"""

import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")
INPUT_SIZE = 640

_session = None


def model_available():
    return os.path.exists(MODEL_PATH)


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        _session = ort.InferenceSession(MODEL_PATH,
                                        providers=["CPUExecutionProvider"])
    return _session


def _letterbox(img, size=INPUT_SIZE):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = (size - nh) / 2, (size - nw) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, left, top


def detect_with_yolo(img, conf_thr=0.25, iou_thr=0.45, top_n=3):
    """يعيد [(score, box4x2)] بنفس صيغة الكشف الكلاسيكي في app.py."""
    if not model_available():
        return []
    sess = _get_session()
    padded, r, left, top = _letterbox(img)
    blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    out = np.squeeze(out)              # (4+nc, 8400)
    if out.shape[0] < out.shape[1]:
        out = out.T                    # (8400, 4+nc)
    scores = out[:, 4:].max(axis=1)
    keep = scores >= conf_thr
    out, scores = out[keep], scores[keep]
    if len(out) == 0:
        return []

    cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    x1 = (cx - bw / 2 - left) / r
    y1 = (cy - bh / 2 - top) / r
    w = bw / r
    h = bh / r
    boxes = np.stack([x1, y1, w, h], axis=1)

    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thr, iou_thr)
    idx = np.array(idx).flatten()
    H, W = img.shape[:2]
    results = []
    for i in idx[:top_n]:
        bx, by, bw_, bh_ = boxes[i]
        xa, ya = max(0, bx), max(0, by)
        xb, yb = min(W - 1, bx + bw_), min(H - 1, by + bh_)
        if xb - xa < 10 or yb - ya < 5:
            continue
        box = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]], dtype="float32")
        results.append((float(scores[i]), box))
    return results
