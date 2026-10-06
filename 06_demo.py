import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

MODEL_PATH = OUTPUT_DIR / "supervised_learning_model.keras"
SCALER_PATH = OUTPUT_DIR / "supervised_learning_scaler.npz"


# =========================================================
# CLASS NAMES
# =========================================================

CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
]


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# =========================================================
# LOAD SCALER
# =========================================================

@st.cache_resource
def load_scaler():

    scaler = np.load(SCALER_PATH)

    mean = scaler["mean"]
    std = scaler["std"]

    std[std == 0] = 1

    return mean, std


model = load_model()
scaler_mean, scaler_std = load_scaler()


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    image = cv2.resize(
        image,
        (128, 128)
    )

    # OpenCV image is BGR
    b, g, r = cv2.split(image)

    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)
    texture = np.std(gray)

    total = mean_r + mean_g + mean_b

    if total > 0:
        red_ratio = mean_r / total
        green_ratio = mean_g / total
        blue_ratio = mean_b / total
    else:
        red_ratio = 0
        green_ratio = 0
        blue_ratio = 0

    features = np.array(
        [
            mean_r,
            mean_g,
            mean_b,
            brightness,
            texture,
            red_ratio,
            green_ratio,
            blue_ratio
        ],
        dtype=np.float32
    )

    return features


# =========================================================
# STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🌱 Plant Disease Classification")

st.write(
    "Upload a plant leaf image to predict its disease class."
)

st.info(
    "AI Approach: Supervised Learning using "
    "a Feed-Forward Neural Network (MLP)"
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "Could not read the uploaded image."
        )

    else:

        # -------------------------------------------------
        # DISPLAY IMAGE
        # -------------------------------------------------

        display_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            display_image,
            caption="Uploaded Leaf Image",
            use_container_width=True
        )


        # -------------------------------------------------
        # EXTRACT FEATURES
        # -------------------------------------------------

        features = extract_features(image)


        # -------------------------------------------------
        # STANDARDIZE FEATURES
        # -------------------------------------------------

        features = (
            features - scaler_mean
        ) / scaler_std


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        probabilities = model.predict(
            features.reshape(1, -1),
            verbose=0
        )[0]


        prediction_index = int(
            np.argmax(probabilities)
        )

        prediction = CLASSES[
            prediction_index
        ]

        confidence = (
            probabilities[prediction_index]
            * 100
        )


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.subheader("Prediction")

        st.success(
            f"🌿 {prediction}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


        # -------------------------------------------------
        # CLASS PROBABILITIES
        # -------------------------------------------------

        st.subheader(
            "Class Probabilities"
        )

        for class_name, probability in zip(
            CLASSES,
            probabilities
        ):

            percentage = probability * 100

            st.write(
                f"**{class_name}**: "
                f"{percentage:.2f}%"
            )

            st.progress(
                float(probability)
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Plant Disease Classification | "
    "Semester 5 AI Project"
)