import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import time
from skimage.feature import hog
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

st.set_page_config(page_title="Fake Face Detector", layout="wide")

# === Load Models ===
@st.cache_resource
def load_models():
    return {
        "Random Forest": joblib.load("models/random_forest_model.pkl"),
        "MobileNetV2": load_model("models/mobilenetv2_fakeface_model.keras"),
        "Xception": load_model("models/xception_fakeface_model.keras"),
        "SVM": joblib.load("models/svm_model.pkl"),
        "Scaler": joblib.load("models/scaler.pkl"),
    }

models = load_models()

# === LBP Function ===
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
    label = "Real" if prediction == 1 else "Fake"
    confidence = models["Random Forest"].predict_proba(hist).max()
    return label, confidence

def predict_with_mobilenet(image):
    resized = cv2.resize(image, (224, 224))
    img_array = resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = models["MobileNetV2"].predict(img_array)[0][0]
    label = "Real" if prediction >= 0.5 else "Fake"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    return label, confidence

def predict_with_xception(image):
    resized = cv2.resize(image, (224, 224))
    img_array = xception_preprocess(resized.astype("float32"))
    img_array = np.expand_dims(img_array, axis=0)
    prediction = models["Xception"].predict(img_array)[0][0]
    label = "Real" if prediction >= 0.5 else "Fake"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    return label, confidence

def predict_with_svm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    lbp_img = lbp(resized)
    hog_features = hog(lbp_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       visualize=False, feature_vector=True)

    scaler = models["Scaler"]
    # Adjust if feature length mismatch due to different HOG output length
    expected_length = scaler.mean_.shape[0]
    if hog_features.shape[0] > expected_length:
        hog_features = hog_features[:expected_length]
    elif hog_features.shape[0] < expected_length:
        hog_features = np.pad(hog_features, (0, expected_length - hog_features.shape[0]), 'constant')

    scaled = scaler.transform([hog_features])
    prediction = models["SVM"].predict(scaled)[0]
    confidence = models["SVM"].predict_proba(scaled).max()
    label = "Real" if prediction == 0 else "Fake"
    return label, confidence


# === Sidebar ===
st.sidebar.title("Model Selector")
model_choice = st.sidebar.selectbox("Choose a Model", (
    "LBP + Random Forest",
    "CNN + MobileNetV2",
    "CNN + Xception",
    "HOG + LBP + SVM"
))

#if 'prediction' in st.session_state:
#    label = st.session_state.prediction
#    conf = st.session_state.confidence
#    st.sidebar.markdown("### 📊 Last Prediction")
#    color = "green" if label == "Real" else "red"
#    st.sidebar.markdown(f"**Result:** <span style='color:{color};font-size:20px;'>{label}</span>", unsafe_allow_html=True)
#    st.sidebar.markdown(f"**Confidence:** {conf * 100:.2f}%", unsafe_allow_html=True)
#    st.sidebar.markdown(f"**Model:** {model_choice}")

# === Main Area ===
st.title("Fake Face Detector")
st.caption("Upload a face photo to determine whether it's real or AI-generated!")

uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="👤 Uploaded Face", channels="BGR", use_container_width=True)

    if st.button("🔍 Predict"):
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
                else:
                    st.error("Model not recognized.")
                    st.stop()

                duration = time.time() - start

                st.session_state.prediction = label
                st.session_state.confidence = conf
                st.session_state.duration = duration

                # Color-coded result
                st.markdown("---")
                st.subheader("🔬 Result")
                if label == "Real":
                    st.success(f"✅ It's a **Real Face** with {conf*100:.2f}% confidence")
                else:
                    st.error(f"❌ It's a **Fake Face** with {conf*100:.2f}% confidence")

                st.caption(f"⏱️ Prediction time: {duration:.2f} seconds")

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")
