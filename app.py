import streamlit as st
from utils.medgpt_pipeline import process_symptom_text
from PIL import Image
import speech_recognition as sr
import tempfile

st.set_page_config(page_title="AI Rural Health Assistant", layout="centered")

st.title("🩺 Rural AI Health Assistant")

st.write("""
Describe your symptoms in any Nigerian language or English, or upload a photo for a preliminary AI-based analysis. 
(Not a replacement for professional medical care.)
""")

tab1, tab2, tab3 = st.tabs(["📝 Text Symptoms", "🖼️ Upload Image", "🎤 Voice Input"])

with tab1:
    user_input = st.text_area("Describe your symptoms:")
    if st.button("Analyze Text"):
        result = process_symptom_text(user_input)
        st.success(result)

with tab2:
    uploaded_file = st.file_uploader("Upload an image of the skin, wound, or eye", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.info("✅ Image received. Inference will run here once models are connected.")
        # TODO: Add model loading and prediction here for MobileNetV2 and YOLOv5

with tab3:
    st.info("Record your symptoms (supports Nigerian languages).")
    audio_file = st.file_uploader("Upload a voice recording (.wav or .mp3)", type=["wav", "mp3"])

    if audio_file:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name

        with sr.AudioFile(tmp_file_path) as source:
            audio_data = recognizer.record(source)
            try:
                transcribed_text = recognizer.recognize_google(audio_data)
                st.success(f"You said: {transcribed_text}")
                response = process_symptom_text(transcribed_text)
                st.success(response)
            except sr.UnknownValueError:
                st.error("Sorry, could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Speech recognition error: {e}")
