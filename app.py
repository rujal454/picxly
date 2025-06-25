import streamlit as st
import cv2
import os
import numpy as np
import pickle
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
import mediapipe as mp
from fer import FER
from ultralytics import YOLO





# ──────────── Config ────────────
st.set_page_config(page_title="🔍 Face Search & YOLOv8 Detection")
st.title("🔍 Face Search  |  😄 Emotion  |  🩹 Blur  |  🎯 YOLOv8")

CLOSED_EYE_THRESHOLD = 0.21
BLUR_THRESHOLD       = 100.0
SIM_THRESHOLD        = 0.50   # 50 % similarity

# ─────────── Model Loading ───────────
@st.cache_resource
def load_face_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

@st.cache_resource
def load_fer():
    return FER(mtcnn=True)

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  # COCO‑80 classes

face_app         = load_face_model()
emotion_detector = load_fer()
yolo_model       = load_yolo()

# MediaPipe face‑mesh (for eye state)
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)

os.makedirs("data/gallery", exist_ok=True)

# ──────── Face‑Search Helpers ────────

def build_index(dir="data/gallery"):
    idx = []
    for fn in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, fn))
        if img is None:
            continue
        for f in face_app.get(img):
            if f.det_score < 0.6:
                continue
            emb = normalize(f.embedding.reshape(1, -1))[0]
            idx.append({"image": fn, "embedding": emb})
    with open("data/index.pkl", "wb") as f:
        pickle.dump(idx, f)
    return len(idx)


def load_index():
    if not os.path.exists("data/index.pkl"):
        return []
    with open("data/index.pkl", "rb") as f:
        return pickle.load(f)


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
    q = normalize(best.embedding.reshape(1, -1))[0]
    sims = [(np.dot(q, e["embedding"]), e["image"]) for e in index]
    sims.sort(reverse=True)
    return best.bbox, sims[:top_k]

# ─────── Eye & Emotion ───────

def detect_emotion(face_rgb):
    res = emotion_detector.detect_emotions(face_rgb)
    if res:
        emo = max(res[0]["emotions"], key=res[0]["emotions"].get)
        return emo, res[0]["emotions"][emo]
    return "Unknown", 0.0


def ear(pts):
    d = lambda a,b: np.linalg.norm(np.array(a)-np.array(b))
    return (d(pts[1],pts[5])+d(pts[2],pts[4]))/(2*d(pts[0],pts[3])) if d(pts[0],pts[3]) else 0


def eyes_closed(img):
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return False
    lm = res.multi_face_landmarks[0]
    l_ids = [33,160,158,133,153,144]; r_ids=[362,385,387,263,373,380]
    h,w = img.shape[:2]
    lp = [(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)) for i in l_ids]
    rp = [(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)) for i in r_ids]
    return (ear(lp)+ear(rp))/2 < CLOSED_EYE_THRESHOLD

# ─────── Blur Helper ───────

def blur_var(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

# ─────── Sidebar ───────
mode = st.sidebar.radio("Mode", ["Face Search", "Object Detection"])

if mode == "Face Search":
    if st.sidebar.button("📂 Build Index"):
        st.sidebar.success(f"Indexed {build_index()} face(s)")
else:
    conf = st.sidebar.slider("Confidence",0.1,1.0,0.5,0.05)
    iou  = st.sidebar.slider("IoU",0.1,1.0,0.5,0.05)

# ─────── FACE SEARCH UI ───────
if mode == "Face Search":
    # File upload
    st.subheader("📁 Upload Image for Face Search")
    up = st.file_uploader("Upload",["jpg","jpeg","png"],key="fs")
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(),np.uint8),1)
        st.image(img,caption="Uploaded",channels="BGR")
        if blur_var(img) < BLUR_THRESHOLD:
            st.warning("🚫 Image looks blurry")
        bbox, matches = search_face(img)
        if bbox is None:
            st.warning("No face detected or index empty.")
        else:
            good = [(s,f) for s,f in matches if s >= SIM_THRESHOLD]
            if not good:
                st.info("No matches ≥ 50 % similarity found.")
            else:
                st.subheader("🟢 Matches ≥ 50 %")
                for s,fname in good:
                    st.image(os.path.join("data/gallery",fname),caption=f"Sim {s:.2f}")
            x1,y1,x2,y2 = map(int,bbox)
            face_rgb = cv2.cvtColor(img[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)
            emo,conf = detect_emotion(face_rgb)
            st.write(f"Emotion: {emo} ({conf:.0%})")
            st.write("👁️ Closed" if eyes_closed(img) else "👁️ Open")

    # Webcam
    st.subheader("📸 Webcam Face Search")
    frame = st.camera_input("Snap", key="camfs")
    if frame:
        img = cv2.cvtColor(np.array(Image.open(frame)), cv2.COLOR_RGB2BGR)
        st.image(img, caption="Captured", channels="BGR")
        if blur_var(img) < BLUR_THRESHOLD:
            st.warning("🚫 Image looks blurry")
        bbox, matches = search_face(img)
        if bbox is None:
            st.warning("No face detected or index empty.")
        else:
            good = [(s,f) for s,f in matches if s >= SIM_THRESHOLD]
            if not good:
                st.info("No matches ≥ 50 % similarity found.")
            else:
                st.subheader("🟢 Matches ≥ 50 %")
                for s,fname in good:
                    st.image(os.path.join("data/gallery",fname),caption=f"Sim {s:.2f}")
            x1,y1,x2,y2 = map(int,bbox)
            face_rgb = cv2.cvtColor(img[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)
            emo,conf = detect_emotion(face_rgb)
            st.write(f"Emotion: {emo} ({conf:.0%})")
            st.write("👁️ Closed" if eyes_closed(img) else "👁️ Open")

# ─────── OBJECT DETECTION UI ───────
if mode == "Object Detection":
    st.subheader("📁 Upload Image for YOLOv8 Detection")
    up = st.file_uploader("Upload",["jpg","jpeg","png"],key="obj")
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(),np.uint8),1)
        res = yolo_model.predict(img,conf=conf,iou=iou)[0]
        st.image(res.plot(),caption="Detections",channels="BGR")
        for b in res.boxes:
            st.write(f"- {yolo_model.names[int(b.cls[0])]}: {float(b.conf[0]):.2f}")

    st.subheader("📸 Webcam Object Detection")
    frame = st.camera_input("Snap", key="camobj")
    if frame:
        img = cv2.cvtColor(np.array(Image.open(frame)), cv2.COLOR_RGB2BGR)
        res = yolo_model.predict(img,conf=conf,iou=iou)[0]
        st.image(res.plot(),caption="Detections",channels="BGR")
        for b in res.boxes:
            st.write(f"- {yolo_model.names[int(b.cls[0])]}: {float(b.conf[0]):.2f}")
