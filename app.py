import os
import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import time
from skimage.feature import hog, local_binary_pattern
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
import gdown
from collections import Counter

st.set_page_config(page_title="Fake Face Detector", layout="wide")

# === Load Models ===
@st.cache_resource
def load_models():
    model_files = {
        "random_forest_model.pkl": "1yMwHbYqPS4jrrjQgao3PFG0polVvmwvg",
        "mobilenetv2_fakeface_model.keras": "1QU38PEDfr11LlrXFfcEf_qiwH8_kaa4M",
        "xception_fakeface_model.keras": "1iZs8YNJmCeQ3XpXj7jy4R_4hjbfgP-at",
        "svm_model.pkl": "1YlsN2Gge21hoBb-ghAH6b3dFunJgdQzr",
        "scaler.pkl": "1rSvFSFd5fJqQGvtkRRR-C7e4Y9YNV-to"
    }

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    for filename, file_id in model_files.items():
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            #st.write(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=True)

    return {
        "Random Forest": joblib.load(os.path.join(model_dir, "random_forest_model.pkl")),
        "MobileNetV2": load_model(os.path.join(model_dir, "mobilenetv2_fakeface_model.keras")),
        "Xception": load_model(os.path.join(model_dir, "xception_fakeface_model.keras")),
        "SVM": joblib.load(os.path.join(model_dir, "svm_model.pkl")),
        "Scaler": joblib.load(os.path.join(model_dir, "scaler.pkl")),
    }

models = load_models()

# === Utility Functions ===
def lbp(image):
    lbp_image = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            center = image[i, j]
            binary = ''.join([
                '1' if image[i-1, j-1] >= center else '0',
                '1' if image[i-1, j] >= center else '0',
                '1' if image[i-1, j+1] >= center else '0',
                '1' if image[i, j+1] >= center else '0',
                '1' if image[i+1, j+1] >= center else '0',
                '1' if image[i+1, j] >= center else '0',
                '1' if image[i+1, j-1] >= center else '0',
                '1' if image[i, j-1] >= center else '0'
            ])
            lbp_image[i, j] = int(binary, 2)
    return lbp_image

# === Prediction Functions ===
def predict_with_random_forest(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    lbp_img = lbp(resized)
    hist, _ = np.histogram(lbp_img.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist.reshape(1, -1)
    prediction = models["Random Forest"].predict(hist)[0]
    confidence = models["Random Forest"].predict_proba(hist).max()
    return ("Real" if prediction == 1 else "Fake", confidence)

def predict_with_mobilenet(image):
    resized = cv2.resize(image, (224, 224))
    img_array = resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = models["MobileNetV2"].predict(img_array)[0][0]
    label = "Real" if prediction >= 0.5 else "Fake"
    confidence = prediction if label == "Real" else 1 - prediction
    return label, confidence

def predict_with_xception(image):
    resized = cv2.resize(image, (224, 224))
    img_array = xception_preprocess(resized.astype("float32"))
    img_array = np.expand_dims(img_array, axis=0)
    prediction = models["Xception"].predict(img_array)[0][0]
    label = "Real" if prediction >= 0.5 else "Fake"
    confidence = prediction if label == "Real" else 1 - prediction
    return label, confidence

def predict_with_svm(image):
    LBP_RADIUS = 1
    LBP_POINTS = 8 * LBP_RADIUS
    LBP_METHOD = 'uniform'

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))

    hog_features = hog(resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       block_norm='L2-Hys', visualize=False, feature_vector=True)

    lbp_image = local_binary_pattern(resized, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)

    features = np.concatenate([hog_features, lbp_hist])
    scaled = models["Scaler"].transform([features])
    prediction = models["SVM"].predict(scaled)[0]
    confidence = models["SVM"].predict_proba(scaled).max()
    return ("Real" if prediction == 0 else "Fake", confidence)

def ensemble_voting(image):
    votes = []
    confidences = []

    for func in [predict_with_random_forest, predict_with_mobilenet, predict_with_xception, predict_with_svm]:
        label, conf = func(image)
        votes.append(label)
        confidences.append(conf)

    final_label = Counter(votes).most_common(1)[0][0]
    avg_conf = sum(confidences) / len(confidences)
    return final_label, avg_conf

# === Sidebar ===
st.sidebar.title("Model Selector")
model_choice = st.sidebar.selectbox("Choose a Model", (
    "LBP + Random Forest",
    "CNN + MobileNetV2",
    "CNN + Xception",
    "HOG + LBP + SVM",
    "Ensemble Voting (All Models)"
))

# === Main Area ===
st.title("Fake Face Detector")
st.caption("Upload a face photo to determine whether it's Real or AI-generated.")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Face", channels="BGR", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            try:
                start = time.time()

                if model_choice == "LBP + Random Forest":
                    label, conf = predict_with_random_forest(image)
                elif model_choice == "CNN + MobileNetV2":
                    label, conf = predict_with_mobilenet(image)
                elif model_choice == "CNN + Xception":
                    label, conf = predict_with_xception(image)
                elif model_choice == "HOG + LBP + SVM":
                    label, conf = predict_with_svm(image)
                elif model_choice == "Ensemble Voting (All Models)":
                    label, conf = ensemble_voting(image)
                else:
                    st.error("Model not recognized.")
                    st.stop()

                duration = time.time() - start

                st.markdown("---")
                st.subheader("Result")
                if label == "Real":
                    st.success(f"It's a **Real Face** with {conf*100:.2f}% confidence")
                else:
                    st.error(f"It's a **Fake Face** with {conf*100:.2f}% confidence")

                st.caption(f"Prediction time: {duration:.2f} seconds")

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")
