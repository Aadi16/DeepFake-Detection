import os
import joblib
from tensorflow.keras.models import load_model

def test_model_loading():
    base_path = "models"

    models_to_test = {
        "Random Forest": os.path.join(base_path, "random_forest_model.pkl"),
        "MobileNetV2": os.path.join(base_path, "mobilenetv2_fakeface_model.keras"),
        "Xception": os.path.join(base_path, "xception_fakeface_model.keras"),
        "SVM": os.path.join(base_path, "svm_model.pkl"),
        "Scaler": os.path.join(base_path, "scaler.pkl"),
    }

    for name, path in models_to_test.items():
        try:
            if path.endswith(".pkl"):
                model = joblib.load(path)
            elif path.endswith(".keras"):
                model = load_model(path)
            print(f"[✓] Loaded {name} successfully.")
        except Exception as e:
            print(f"[✗] Failed to load {name}: {e}")

if __name__ == "__main__":
    test_model_loading()
