import streamlit as st
from utils.medgpt_pipeline import process_symptom_text
from PIL import Image
from streamlit_mic_recorder import mic_recorder
import tempfile
import requests
import folium
from streamlit_folium import st_folium

from streamlit_javascript import st_javascript

st.set_page_config(page_title="AI Health Assistant", layout="centered")

# Get user's geolocation from browser
coords = st_javascript("""
navigator.geolocation.getCurrentPosition(
    (pos) => {
        const coords = { lat: pos.coords.latitude, lng: pos.coords.longitude };
        window.localStorage.setItem("coords", JSON.stringify(coords));
    },
    (err) => { console.error("Geolocation error:", err); }
);
JSON.parse(window.localStorage.getItem("coords") || "{}")
""")

if isinstance(coords, dict):
    lat = coords.get("lat", 6.5244)
    lng = coords.get("lng", 3.3792)
else:
    lat, lng = 6.5244, 3.3792

user_location = (lat, lng)  # fallback: Lagos
st.session_state.setdefault("user_coords", {"lat": user_location[0], "lng": user_location[1]})
st.session_state.setdefault("show_map", True)

st.title("🩺 AI Health Assistant")

# --- Dataset Selector ---
dataset_choice = st.selectbox("Reference Dataset (optional)", ["usmle", "afri"])

# --- Text Symptom Input ---
st.subheader("🔊 Describe Your Symptoms")
with st.form("symptom_form"):
    user_input = st.text_area("Type your symptoms:")
    submit_text = st.form_submit_button("🧬 Analyze Symptoms")

if submit_text and user_input:
    with st.spinner("🧠 Thinking..."):
        result, top_pharmacies = process_symptom_text(user_input, dataset=dataset_choice)
        st.markdown("### 💡 AI Advice")
        st.success(result)

        st.session_state["ai_response"] = result
        st.session_state["map_results"] = top_pharmacies
        if top_pharmacies:
            st.session_state["location_coords"] = (top_pharmacies[0]["lat"], top_pharmacies[0]["lng"])

# --- Voice Input ---
st.subheader("🎤 Or Record Your Voice")
audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹️ Stop Recording", key="voice_input")

if audio:
    st.success("✅ Recording complete!")
    st.audio(audio['bytes'], format="audio/wav")

    if st.button("🧠 Transcribe & Analyze Voice"):
        with st.spinner("🪄 Transcribing with Whisper via Groq..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    tmp_audio.write(audio['bytes'])
                    tmp_path = tmp_audio.name

                headers = {"Authorization": f"Bearer {st.secrets.get('GROQ_API_KEY', 'YOUR_FALLBACK_KEY')}"}
                files = {"file": open(tmp_path, "rb"), "model": (None, "whisper-large-v3")}
                response = requests.post("https://api.groq.com/openai/v1/audio/transcriptions", headers=headers, files=files)

                if response.status_code == 200:
                    text = response.json().get("text", "")
                    st.markdown(f"🗣️ **You said:** `{text}`")

                    if text:
                        with st.spinner("🔬 Analyzing symptoms..."):
                            result, top_pharmacies = process_symptom_text(text, dataset=dataset_choice)
                            st.success(result)
                            st.session_state["ai_response"] = result
                else:
                    st.error(f"❌ Transcription failed: {response.text}")
            except Exception as e:
                st.error(f"🚨 Error during transcription: {e}")

if st.button("🔁 Retake Recording"):
    st.rerun()

# --- Image Input ---
st.subheader("🖼️ Photo of Affected Area")
image = st.camera_input("Take a photo") or st.file_uploader("Or upload an image", type=["jpg", "png"])

if image:
    st.image(Image.open(image), caption="Input Image", use_column_width=True)
    st.info("✅ Image received. Inference will run here once models are connected.")

# --- Single Final Map ---
st.markdown("### 🗺️ Nearby Medical Help")

map_center = st.session_state.get("location_coords", user_location)
m = folium.Map(location=map_center, zoom_start=14)

if st.session_state.get("map_results"):
    for place in st.session_state["map_results"]:
        folium.Marker(
            [place["lat"], place["lng"]],
            popup=place["name"],
            tooltip=place["name"],
            icon=folium.Icon(color="green", icon="plus")
        ).add_to(m)
else:
    folium.Marker(
        location=map_center,
        tooltip="📍 You are here",
        icon=folium.Icon(color="blue")
    ).add_to(m)

st_folium(m, width=700, height=500)
