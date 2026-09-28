
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from model_components import CBAM, FocalLoss

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "improved_model.keras"

CLASS_NAMES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
]

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="centered",
)

st.title("🌱 Plant Disease Classifier")
st.write(
    "Upload a tomato or potato leaf image. "
    "The model predicts the leaf's disease class."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Improved model not found. "
            "Run 04_improved_model.py first."
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "CBAM": CBAM,
            "FocalLoss": FocalLoss,
        },
        compile=False,
    )


uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(
            image,
            caption="Uploaded leaf",
            use_container_width=True,
        )

        if st.button("Predict disease", type="primary"):
            with st.spinner("Analyzing the leaf..."):
                model = load_model()

                img = image.resize((224, 224))
                arr = np.asarray(img, dtype=np.float32)
                arr = np.expand_dims(arr, axis=0)

                # The model already preprocesses its input.
                # Pass raw RGB pixel values (0–255).
                probs = model.predict(arr, verbose=0)[0]

                top_indices = np.argsort(probs)[::-1][:3]

            best = int(top_indices[0])
            predicted_class = CLASS_NAMES[best].replace(
                "___", " — "
            ).replace("_", " ")

            st.success(f"Prediction: **{predicted_class}**")
            st.metric(
                "Model confidence",
                f"{probs[best] * 100:.2f}%",
            )

            st.subheader("Top 3 predictions")

            for idx in top_indices:
                label = CLASS_NAMES[int(idx)].replace(
                    "___", " — "
                ).replace("_", " ")

                st.write(
                    f"**{label}** — {probs[idx] * 100:.2f}%"
                )

            st.caption(
                "Predictions are model estimates and may "
                "not always be correct."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
