import streamlit as st
import cv2
import os
import numpy as np
import pickle
import zipfile
from PIL import Image
from insightface.app import FaceAnalysis
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from typing import Optional, Tuple
from ultralytics import YOLO

"""
Picxly 

"""

# ─── Config ───────────────────────────────────────────────────
st.set_page_config(page_title="🔍 Picxly – Face & Object Search")


CLOSED_EYE_THRESHOLD = 0.21
BLUR_THRESHOLD = 60.0
SIM_THRESHOLD = 0.50      # webcam 50 %
PERFECT_THRESHOLD = 0.9999 # file‑upload 100 %

# ─── Model Loading (cached) ───────────────────────────────────
@st.cache_resource
def load_face_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))  # CPU
    return app


@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

face_app = load_face_model()
yolo_model = load_yolo()

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)

os.makedirs("data/gallery", exist_ok=True)

# ─── Optional Emotions (safe import) ─────────────────────────────
@st.cache_resource
def load_fer_optional():
    try:
        from fer import FER  # type: ignore
        return FER(mtcnn=True)
    except Exception:
        return None

emotion_detector_opt = load_fer_optional()

# Lightweight TF-free emotion based on Face Mesh landmarks
def _euclid(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def _mesh_points(image_rgb, indices):
    res = face_mesh.process(image_rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0]
    h, w = image_rgb.shape[:2]
    pts = []
    for i in indices:
        p = lm.landmark[i]
        pts.append((p.x * w, p.y * h))
    return pts

def detect_emotion_lite(face_rgb) -> Optional[Tuple[str, float]]:
    # Indices: mouth corners (61, 291), upper/lower lip (13, 14)
    idx = [61, 291, 13, 14]
    pts = _mesh_points(face_rgb, idx)
    if pts is None:
        return None
    left, right, up, down = pts
    face_w = float(face_rgb.shape[1]) if face_rgb is not None else 1.0
    if face_w <= 0:
        face_w = 1.0
    mouth_width = _euclid(left, right)
    mouth_open = _euclid(up, down)
    # Ratios normalized by face width for robustness
    open_ratio = mouth_open / max(1e-6, face_w)
    smile_ratio = mouth_width / max(1e-6, face_w)

    # Heuristic thresholds (empirical)
    if open_ratio > 0.12:
        return ("Surprised", min(1.0, (open_ratio - 0.12) / 0.15))
    if smile_ratio > 0.42 and open_ratio < 0.08:
        return ("Happy", min(1.0, (smile_ratio - 0.42) / 0.20))
    return ("Neutral", 0.5)

# Optional emotion detection wrapper (prefers FER if installed, else lite)
def detect_emotion_optional(face_rgb) -> Optional[Tuple[str, float]]:
    if emotion_detector_opt is not None:
        try:
            res = emotion_detector_opt.detect_emotions(face_rgb)
            if res:
                emo = max(res[0]["emotions"], key=res[0]["emotions"].get)
                return emo, float(res[0]["emotions"][emo])
        except Exception:
            pass
    return detect_emotion_lite(face_rgb)

# ─── Index Helpers ─────────────────────────────────────────

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)

def build_index_from_folder(folder="data/gallery"):
    idx = []
    for fn in os.listdir(folder):
        img_path = os.path.join(folder, fn)
        img = cv2.imread(img_path)
        if img is None:
            continue
        for f in face_app.get(img):
            if f.det_score < 0.6:
                continue
            emb = l2_normalize(f.embedding)
            idx.append({"image": fn, "embedding": emb})
    with open("data/index.pkl", "wb") as f:
        pickle.dump(idx, f)
    return len(idx)

def load_index():
    if not os.path.exists("data/index.pkl"):
        return []
    with open("data/index.pkl", "rb") as f:
        return pickle.load(f)

# ─── Face Search ────────────────────────────────────────

def search_face(img: np.ndarray, top_k=5):
    index = load_index()
    if not index:
        return None, []
    faces = face_app.get(img)
    if not faces:
        return None, []
    best = max(faces, key=lambda f: f.det_score)
    if best.det_score < 0.6:
        return None, []
    q = l2_normalize(best.embedding)
    sims = [(np.dot(q, e["embedding"]), e["image"]) for e in index]
    sims.sort(reverse=True)
    return best.bbox, sims[:top_k]

# ─── Blink helpers ─────────────────────────────────

def ear(pts):
    d = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))
    return (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2 * d(pts[0], pts[3])) if d(pts[0], pts[3]) else 0

def eyes_closed(img):
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return False
    lm = res.multi_face_landmarks[0]
    l_ids = [33, 160, 158, 133, 153, 144]
    r_ids = [362, 385, 387, 263, 373, 380]
    h, w = img.shape[:2]
    lp = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in l_ids]
    rp = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in r_ids]
    return (ear(lp) + ear(rp)) / 2 < CLOSED_EYE_THRESHOLD

# ─── Misc util ─────────────────────────────────

def blur_var(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize size to make variance comparable across crops
    h, w = gray.shape[:2]
    target = 256
    if min(h, w) > 0 and min(h, w) < target:
        scale = target / float(min(h, w))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ─── SIDEBAR ────────────────────────────────
mode = st.sidebar.radio("Mode", ["Face Search", "Object Detection"])

# Admin upload faces
aifiles = None
if mode == "Face Search":
    st.sidebar.subheader("🛠 Admin – Upload Faces (.zip or images)")
    aifiles = st.sidebar.file_uploader("Upload", type=["zip", "jpg", "jpeg", "png"], accept_multiple_files=True)
    if aifiles and st.sidebar.button("Build Index"):
        for file in aifiles:
            if file.name.endswith(".zip"):
                with zipfile.ZipFile(file) as z:
                    z.extractall("data/gallery")
            else:
                with open(os.path.join("data/gallery", file.name), "wb") as f:
                    f.write(file.read())
        st.sidebar.success(f"Indexed {build_index_from_folder()} face(s)")

# ─── FACE SEARCH UI ─────────────────────────────────
if mode == "Face Search":
    st.subheader("📁 Upload Image for 100 % Face Match")
    up = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], key="fs")
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        st.image(img, caption="Uploaded", channels="BGR")
        # Always run search; only warn if blurry (do not stop)
        bbox, matches = search_face(img)
        if bbox is None:
            st.warning("No face detected or index empty.")
        else:
            # Compute blur on detected face crop if possible; otherwise on full image
            x1, y1, x2, y2 = map(int, bbox)
            crop = img[y1:y2, x1:x2]
            bv = blur_var(crop if crop.size else img)
            if bv < BLUR_THRESHOLD:
                st.info(f"⚠️ Image looks blurry (variance {bv:.1f} < {BLUR_THRESHOLD:.1f}). Results may be less accurate.")
            perfect = [(s, f) for s, f in matches if s >= PERFECT_THRESHOLD]
            if perfect:
                st.subheader("🟢 100 % Identical Faces")
                for s, fname in perfect:
                    st.image(os.path.join("data/gallery", fname), caption=f"Sim {s:.4f}")
            else:
                st.info("No exact matches.")
            st.write(":eye: Closed" if eyes_closed(img) else ":eye: Open")
            if st.checkbox("Show emotion", value=False, key="emo_upload"):
                face_rgb = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                emo = detect_emotion_optional(face_rgb)
                if emo is None:
                    st.info("Emotion model not installed. In venv: pip install fer moviepy imageio-ffmpeg")
                else:
                    name, conf = emo
                    st.write(f"Emotion: {name} ({conf:.0%})")

    st.subheader("📸 Webcam Face Search (≥ 50 %)")
    frame = st.camera_input("Snap", key="camfs")
    if frame:
        img = cv2.cvtColor(np.array(Image.open(frame)), cv2.COLOR_RGB2BGR)
        st.image(img, caption="Captured", channels="BGR")
        bbox, matches = search_face(img)
        if bbox is not None and matches:
            good = [(s, f) for s, f in matches if s >= SIM_THRESHOLD]
            if good:
                st.subheader("🟢 Matches ≥ 50 %")
                for s, fname in good:
                    st.image(os.path.join("data/gallery", fname), caption=f"Sim {s:.2f}")
            x1, y1, x2, y2 = map(int, bbox)
            st.write(":eye: Closed" if eyes_closed(img) else ":eye: Open")

    st.subheader("📂 Batch Analyze (multi-file)")
    batch_files = st.file_uploader("Drop multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch")
    if batch_files:
        imgs = []
        metas = []
        for f in batch_files:
            arr = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(arr, 1)
            if img is None:
                continue
            imgs.append(img)
            metas.append(f.name)
        if not imgs:
            st.warning("No valid images.")
        else:
            cols = st.columns(3)
            for i, (img, name) in enumerate(zip(imgs, metas)):
                bbox, matches = search_face(img)
                eye_txt = ":eye: Closed" if eyes_closed(img) else ":eye: Open"
                sim_txt = "No match"
                if matches:
                    s, fname = matches[0]
                    sim_txt = f"Top match {s:.2f} → {fname}"
                with cols[i % 3]:
                    st.image(img, caption=name, channels="BGR")
                    st.caption(f"{sim_txt} · {eye_txt}")
            if st.checkbox("Show emotion (webcam)", value=False, key="emo_cam"):
                face_rgb = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                emo = detect_emotion_optional(face_rgb)
                if emo is None:
                    st.info("Emotion model not installed. In venv: pip install fer moviepy imageio-ffmpeg")
                else:
                    name, conf = emo
                    st.write(f"Emotion: {name} ({conf:.0%})")

# ─── OBJECT DETECTION ─────────────────────────────────
if mode == "Object Detection":
    st.subheader("📁 Upload Image for Object Detection")
    obj_up = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], key="obj")
    if obj_up:
        img = cv2.imdecode(np.frombuffer(obj_up.read(), np.uint8), 1)
        res = yolo_model.predict(img)[0]
        for r in res.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cls = int(r.cls[0])
            label = yolo_model.names[cls]
            conf = r.conf[0].item()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        st.image(img, caption="Detected Objects", channels="BGR")

    st.subheader("📸 Webcam Object Detection")
    obj_cam = st.camera_input("Snap", key="camobj")
    if obj_cam:
        img = cv2.cvtColor(np.array(Image.open(obj_cam)), cv2.COLOR_RGB2BGR)
        res = yolo_model.predict(img)[0]
        for r in res.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cls = int(r.cls[0])
            label = yolo_model.names[cls]
            conf = r.conf[0].item()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        st.image(img, caption="Detected Objects", channels="BGR")
