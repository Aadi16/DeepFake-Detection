**Deepfake Image Detection**

**1. Objective:**

The primary objective of this project is to detect deepfake images — artificially generated or manipulated facial images — using machine learning and deep learning techniques. As deepfakes pose a serious threat to digital media authenticity, our system is aimed at providing a reliable and efficient way to detect such content.

**2. Dataset:**

**Dataset Name:** DeepFake Detection Challenge (DFDC)
**Source:** Kaggle – DFDC Dataset
**Contents:** Real and manipulated face video clips
**Processing:** We extracted frames from videos and stored them in two folders: real/ and fake/.

**2.1. Preprocessing Steps:**

Face detection (via OpenCV)
Resizing to 224x224
Grayscale conversion (for LBP)
Normalization between [0, 1]

**3. Approaches Used:**

We explored and compared four different techniques:
HOG + LBP + SVM Classifier
CNN + MobilenetV2
CNN + Xception models
LBP+RandomForest Model

**Approach 1:  HOG + LBP + SVM Classifier [Aaditya Singh]**

I implemented a traditional machine learning pipeline that combined Histogram of Oriented Gradients (HOG) and Local Binary Pattern (LBP) feature extraction with an SVM classifier. To fine-tune the model, I used GridSearchCV for hyperparameter optimization and reached an accuracy of 83%. This approach was valuable for benchmarking classical techniques against deep learning-based solutions.

**Framework Used:**
Language: Python
Libraries: OpenCV, scikit-image, scikit-learn, NumPy, joblib

**Architecture Overview:**
Data Preprocessing:
Loaded images from the dataset (real and fake).
Resized all images to a fixed size of 128x128 pixels.
Converted images to grayscale for feature extraction.

**Feature Extraction:**
HOG (Histogram of Oriented Gradients): Captures edge and shape structure from images.
LBP (Local Binary Pattern): Extracts texture features.
Combined both HOG and LBP features into a single feature vector.

**Feature Normalization:**
Applied StandardScaler to normalize features and improve SVM performance.

**Model Training**
Split the dataset into training and testing sets (80:20).
Used Support Vector Machine (SVM) with RBF kernel.
Performed GridSearchCV for hyperparameter tuning (C, gamma).

**Model Evaluation:**
Evaluated performance using a classification report (precision, recall, F1-score, accuracy).

**Model Saving:**
Saved trained model (svm_model.pkl) and scaler (scaler.pkl) for deployment.

**Approach 2:  CNN + MobilenetV2 [Mayank Goyal]**

For this model, I utilized MobileNetV2 as a backbone for feature extraction by freezing its pretrained layers to enable effective transfer learning. The input images were normalized and enhanced through data augmentation techniques like random flips and rotations to increase robustness. On top of the base model, I added a lightweight classifier tailored for binary classification (real vs. fake). The training process was optimized using binary cross-entropy loss, along with early stopping and model checkpointing to ensure stability and prevent overfitting.

**Framework Used:**
TensorFlow/Keras—used for model building, training, and evaluation.
Pretrained MobileNetV2—used for transfer learning with ImageNet weights.

**Architecture Overview:**
This architecture leverages transfer learning with a lightweight MobileNetV2 backbone to extract high-level features, while data augmentation improves generalization and prevents overfitting. The final layers reduce overfitting and perform binary classification using a sigmoid activation.

1. Input Layer (224x224x3)
  Takes RGB images resized to 224x224 pixels.
  Defines the expected input shape for MobileNetV2.
2. Data Augmentation Layer (Flip, Rotate, Zoom)
  Randomly flips, rotates, and zooms images to increase data variety.
  Improves generalization and helps prevent overfitting.
3. Pretrained Base Model: MobileNetV2
  Loaded with ImageNet weights to use learned visual features.
  Initially frozen to act as a feature extractor, later fine-tuned.
4. GlobalAveragePooling2D
  Reduces feature maps to a 1D vector of 1280 values.
  Decreases the number of parameters and risk of overfitting.
5. Output Layer (1 unit, sigmoid activation)
  Outputs a probability between 0 (fake) and 1 (real).
  Suitable for binary classification problems like real vs. fake face detection.

**Approach 3:  CNN + Xception models [Suraj Mourya]**

In this project, I implemented a binary image classification pipeline to distinguish between real and fake images using the Xception model. I started by using Xception as a fixed feature extractor to leverage its pretrained capabilities, then fine-tuned the upper layers to better adapt it to our specific dataset. The workflow also includes custom image preprocessing, normalization, and the use of callbacks like early stopping and model checkpoints to monitor and improve training performance.

**Framework Used:**

Deep Learning Framework: TensorFlow + Keras
Pre-trained Model: Xception (with ImageNet weights)
Other Libraries: NumPy, Matplotlib, OpenCV, PIL (for visualization and image handling)

**Architecture Overview:**

Input Image (224x224x3)
Xception (Pre-trained, no top, frozen initially)
GlobalAveragePooling2D
Dropout (0.3)
Dense Layer (256 units, ReLU)
Dense Output Layer (1 unit, Sigmoid)
Binary Prediction (Real or Fake)

**Approach 4:  LBP+RandomForest Model [Jatin Shrivas]**

I actively contributed to the development of a traditional computer vision-based real vs. fake face detection system using Local Binary Pattern (LBP) feature extraction and Random Forest classification. I implemented the image preprocessing pipeline, LBP-based feature engineering, model training, and evaluation. I also developed the prediction module and handled model serialization for deployment. My efforts helped achieve a classification accuracy of 74% on the test dataset.

**Methodology:**
Traditional computer vision approach based on handcrafted feature extraction (Local Binary Pattern) and classical machine learning classification (Random Forest).

**System Architecture:**

**Input Data:** The system uses a labeled dataset containing real and fake face images.
**Preprocessing:** Images are converted to grayscale and resized to a fixed size to ensure uniformity.
**Feature Extraction:** LBP is applied to extract texture-based features from each image, which are then converted into histograms.
**Model Training:** The feature vectors are split into training and testing sets. A random forest classifier is trained to distinguish between real and fake faces.
**Evaluation:** The trained model is evaluated using precision, recall, the F1 score, and a confusion matrix to measure accuracy.
**Model Deployment:** The trained model is saved using joblib and can be loaded to perform real-time predictions on new images.

**DeepFake Detection Website:**

This Streamlit app is built to detect whether a face image is real or generated by AI. I’ve integrated several models—Random Forest, MobileNetV2, Xception, and SVM—each using unique feature extraction techniques such as LBP and HOG. Once the user uploads an image and selects a model, the app runs the prediction and displays the result along with a confidence score. To keep things efficient, all models are preloaded for quicker response time.

**Website Link:** https://deepfake-detection-app.streamlit.app/
