import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 AI-Based Plant Disease Detection System")
st.write(
    "Upload a plant leaf photograph to identify potential "
    "diseases using our trained MobileNetV2 model."
)

st.info("Supported plants: Tomato, Potato and Bell Pepper.")

# --------------------------------------------------
# 2. LOAD TRAINED MODEL AND CLASS NAMES
# --------------------------------------------------

MODEL_PATH = "models/plant_disease_mobilenetv2_final.keras"
CLASS_NAMES_PATH = "models/class_names.json"


@st.cache_resource
def load_model():
    trained_model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as file:
        labels = json.load(file)

    return trained_model, labels


try:
    model, class_names = load_model()
    st.success("MobileNetV2 model loaded successfully!")
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.info(
        "Please ensure the model and class_names.json "
        "files are uploaded to the correct repository folders."
    )
    st.stop()


# --------------------------------------------------
# 3. DISEASE INFORMATION DATABASE
# --------------------------------------------------

disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "status": "Disease Detected",
        "description": "Bacterial spot is a bacterial disease affecting pepper leaves and fruits.",
        "symptoms": "Small, dark, water-soaked spots on leaves, sometimes with yellow edges.",
        "care": "Remove severely infected leaves, avoid overhead watering, maintain good spacing and seek local agricultural advice."
    },
    "Pepper__bell___healthy": {
        "plant": "Bell Pepper",
        "status": "Healthy",
        "description": "The leaf is classified as healthy, with no targeted disease detected by the model.",
        "symptoms": "No obvious disease symptoms identified.",
        "care": "Maintain proper watering, balanced nutrition and regular plant monitoring."
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "status": "Disease Detected",
        "description": "Early blight is a fungal disease commonly associated with Alternaria species.",
        "symptoms": "Brown spots with concentric rings, often surrounded by yellowing tissue.",
        "care": "Remove infected foliage, avoid wetting leaves, rotate crops and seek local agricultural advice."
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "status": "Disease Detected",
        "description": "Late blight is a serious disease caused by Phytophthora infestans.",
        "symptoms": "Dark, water-soaked lesions that may spread rapidly.",
        "care": "Remove infected plant material safely, avoid overhead irrigation and seek prompt agricultural advice."
    },
    "Potato___healthy": {
        "plant": "Potato",
        "status": "Healthy",
        "description": "The leaf is classified as healthy, with no targeted disease detected by the model.",
        "symptoms": "No obvious disease symptoms identified.",
        "care": "Continue proper watering, balanced fertilization and regular monitoring."
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomato",
        "status": "Disease Detected",
        "description": "Bacterial spot is a disease that can affect tomato leaves, stems and fruits.",
        "symptoms": "Small dark spots on leaves, sometimes with yellow halos.",
        "care": "Remove severely affected foliage, avoid handling wet plants and maintain good spacing."
    },
    "Tomato_Early_blight": {
        "plant": "Tomato",
        "status": "Disease Detected",
        "description": "Early blight is a fungal disease that commonly affects older tomato leaves.",
        "symptoms": "Brown lesions with target-like concentric rings and surrounding yellow areas.",
        "care": "Remove infected lower leaves, use mulch to reduce soil splash and rotate crops."
    },
    "Tomato_Late_blight": {
        "plant": "Tomato",
        "status": "Disease Detected",
        "description": "Late blight is a potentially destructive disease caused by Phytophthora infestans.",
        "symptoms": "Irregular dark lesions that may spread quickly, sometimes with pale growth in humid conditions.",
        "care": "Remove infected plant material, avoid overhead watering and seek prompt agricultural advice."
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomato",
        "status": "Disease Detected",
        "description": "Tomato leaf mold is a fungal disease favored by high humidity.",
        "symptoms": "Yellow patches on the upper leaf surface and olive-green or brown growth underneath.",
        "care": "Improve ventilation, reduce leaf wetness and manage humidity."
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomato",
        "status": "Disease Detected",
        "description": "Septoria leaf spot is a fungal disease caused by Septoria lycopersici.",
        "symptoms": "Small circular spots with pale centers and dark borders, often appearing on lower leaves first.",
        "care": "Remove affected lower leaves, avoid splashing water onto foliage, use mulch and rotate crops."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomato",
        "status": "Pest Detected",
        "description": "Two-spotted spider mites are tiny pests that feed on plant leaf cells.",
        "symptoms": "Fine pale speckling, yellowing, bronzing and sometimes delicate webbing on leaves.",
        "care": "Inspect the undersides of leaves, remove heavily affected foliage and seek suitable pest-management advice."
    },
    "Tomato__Target_Spot": {
        "plant": "Tomato",
        "status": "Disease Detected",
        "description": "Target spot is a fungal disease that can affect tomato foliage and fruit.",
        "symptoms": "Brown lesions with concentric rings that may enlarge and merge.",
        "care": "Remove infected leaves, improve airflow, avoid prolonged leaf wetness and rotate crops."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "plant": "Tomato",
        "status": "Disease Detected",
        "description": "Tomato yellow leaf curl is a viral disease commonly transmitted by whiteflies.",
        "symptoms": "Upward leaf curling, yellowing, stunted growth and reduced plant vigor.",
        "care": "Monitor whiteflies, control weeds and seek local agricultural guidance."
    },
    "Tomato__Tomato_mosaic_virus": {
        "plant": "Tomato",
        "status": "Disease Detected",
        "description": "Tomato mosaic virus is a viral disease that can affect tomato growth and leaf appearance.",
        "symptoms": "Mottled light and dark green patterns, leaf distortion and sometimes reduced growth.",
        "care": "Clean tools, wash hands after handling plants and use healthy planting material."
    },
    "Tomato_healthy": {
        "plant": "Tomato",
        "status": "Healthy",
        "description": "The leaf is classified as healthy, with no targeted disease detected by the model.",
        "symptoms": "No obvious disease symptoms identified.",
        "care": "Maintain consistent watering, balanced nutrition and regular inspection."
    }
}


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    image_array = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )
    )

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# 5. DISEASE PREDICTION
# --------------------------------------------------

def predict_disease(image):
    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_disease = class_names[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    top_indices = np.argsort(predictions)[-3:][::-1]

    top_predictions = [
        (class_names[i], float(predictions[i]) * 100)
        for i in top_indices
    ]

    return predicted_disease, confidence, top_predictions


# --------------------------------------------------
# 6. IMAGE UPLOAD AND ANALYSIS
# --------------------------------------------------

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("📸 Upload Leaf Image")

    uploaded_file = st.file_uploader(
        "Choose a leaf photograph",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded Leaf Image",
            use_container_width=True
        )

with right:
    st.subheader("🧠 AI Prediction Results")

    if uploaded_file is not None:

        if st.button("🔍 Analyze Leaf", type="primary"):

            with st.spinner("Analyzing leaf image..."):

                predicted_disease, confidence, top_predictions = (
                    predict_disease(image)
                )

                info = disease_info[predicted_disease]

                st.session_state["latest_result"] = {
                    "disease": predicted_disease,
                    "confidence": confidence,
                    "info": info,
                    "top_predictions": top_predictions
                }

                if "history" not in st.session_state:
                    st.session_state["history"] = []

                st.session_state["history"].append({
                    "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    "Plant": info["plant"],
                    "Prediction": predicted_disease.replace("_", " "),
                    "Confidence": f"{confidence:.2f}%"
                })

    result = st.session_state.get("latest_result")

    if result:
        info = result["info"]

        st.markdown(
            f"### {result['disease'].replace('_', ' ')}"
        )

        st.write(f"**Plant:** {info['plant']}")
        st.write(f"**Status:** {info['status']}")

        st.metric(
            "Prediction Confidence",
            f"{result['confidence']:.2f}%"
        )

        st.progress(
            min(result["confidence"] / 100, 1.0)
        )

        st.write("**Description**")
        st.write(info["description"])

    else:
        st.info("Upload a leaf image and click Analyze Leaf.")


# --------------------------------------------------
# 7. DISEASE INFORMATION
# --------------------------------------------------

result = st.session_state.get("latest_result")

if result:
    st.divider()
    st.header("🌱 Disease Information")

    info = result["info"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔎 Symptoms")
        st.write(info["symptoms"])

    with col2:
        st.subheader("💚 Care Recommendations")
        st.write(info["care"])

    st.divider()
    st.header("📊 Top 3 Model Predictions")

    for name, score in result["top_predictions"]:
        st.write(
            f"**{name.replace('_', ' ')} — {score:.2f}%**"
        )
        st.progress(min(score / 100, 1.0))


# --------------------------------------------------
# 8. PREDICTION HISTORY
# --------------------------------------------------

st.divider()
st.header("📜 Prediction History")

history = st.session_state.get("history", [])

if history:
    st.dataframe(
        history,
        use_container_width=True
    )

    if st.button("Clear Prediction History"):
        st.session_state["history"] = []
        st.rerun()
else:
    st.write("No predictions have been recorded in this session.")


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Model: MobileNetV2 Transfer Learning | "
    "Input size: 224 × 224 | "
    "Classification categories: 15"
)

st.warning(
    "Disclaimer: This application provides educational predictions, "
    "not professional agricultural diagnoses. It is trained on "
    "Pepper, Potato and Tomato leaf categories. Results may be "
    "unreliable for other plants or images unlike the training data."
)

