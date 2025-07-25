import streamlit as st
from utils.medgpt_pipeline import process_symptom_text
from PIL import Image
from streamlit_mic_recorder import mic_recorder
import tempfile
import requests
import folium
from streamlit_folium import st_folium
from utils.image_inference import detect_skin_condition



from streamlit_javascript import st_javascript


# ---------- CONFIG ----------
st.set_page_config(page_title="AI Health Assistant", layout="centered")
st.title("🩺 AI Health Assistant")

# ---------- LANGUAGE / TTS ----------
def speak_text(text):
    with st.spinner("🎧 Generating speech..."):
        try:
            headers = {
                "Authorization": f"Bearer {st.secrets.get('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "playai-tts",
                "input": text,
                "voice": "Arista-PlayAI"
            }

            response = requests.post(
                "https://api.groq.com/openai/v1/audio/speech",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                    tmp_audio.write(response.content)
                    st.audio(tmp_audio.name, format="audio/mp3")
                    st.success("🔊 Playing audio...")
            else:
                st.error(f"❌ TTS failed: {response.text}")
        except Exception as e:
            st.error(f"🚨 Error during TTS: {e}")     


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


# --- Dataset Selector ---
dataset_choice = st.selectbox("Reference Dataset (optional)", ["usmle", "afri"])

# --- Text Symptom Input ---
st.subheader("🔊 Describe Your Symptoms")
with st.form("symptom_form"):
    user_input = st.text_area("Type your symptoms:")
    submit_text = st.form_submit_button("🧬 Analyze Symptoms")

if submit_text and user_input:
    with st.spinner("🧠 Thinking..."):
        result, _ = process_symptom_text(user_input, dataset=dataset_choice)  # Process the input text
        st.markdown("### 💡 AI Advice from Text Input")
        st.success(result)

        st.session_state["text_response"] = result

if st.session_state.get("text_response"):
    if st.button("🔊 Read Aloud", key="read_text"):
        speak_text(st.session_state["text_response"])  



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

                headers = {
                    "Authorization": f"Bearer {st.secrets.get('GROQ_API_KEY', 'YOUR_FALLBACK_KEY')}"
                }
                files = {
                    "file": open(tmp_path, "rb"),
                    "model": (None, "whisper-large-v3")
                }

                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files
                )

                if response.status_code == 200:
                    text = response.json().get("text", "")
                    st.markdown(f"🗣️ **You said:** `{text}`")

                    if text:
                        with st.spinner("🔬 Analyzing symptoms..."):
                            result, _ = process_symptom_text(text, dataset=dataset_choice)
                            st.session_state["voice_response"] = result
                            st.markdown("💡 **AI Advice from Voice Input:**")
                            st.success(result)
                    else:
                        st.warning("⚠️ No speech detected.")
                else:
                    st.error(f"❌ Transcription failed: {response.text}")

            except Exception as e:
                st.error(f"🚨 Error during transcription: {e}")

if st.session_state.get("voice_response"):
    if st.button("🔊 Read Aloud", key="read_voice"):
        speak_text(st.session_state["voice_response"])

# 🔁 Retake recording: clears AI result and re-runs app
if st.button("🔁 Retake Recording"):
    st.session_state.pop("ai_response", None)
    st.rerun()

# --- Image Input + Detection + MedGPT Advice ---
from utils.image_inference import detect_skin_condition
from utils.medgpt_pipeline import process_symptom_text
from PIL import Image
import streamlit as st
import folium
from streamlit_folium import st_folium

st.subheader("🖼️ Photo of Affected Area")
image = st.camera_input("Take a photo") or st.file_uploader("Or upload an image", type=["jpg", "png"])

if image:
    img_pil = Image.open(image).convert("RGB")
    st.image(img_pil, caption="📷 Original Input", use_container_width=True)

    with st.spinner("🔍 Running visual diagnosis..."):
        try:
            source, vision_outputs = detect_skin_condition(image.getvalue())
            st.markdown(f"✅ Vision model used: `{source}`")

            for result in vision_outputs:
                st.markdown("### 🧠 Visual Analysis")
                st.info(result["query"])

                st.markdown("### 🧬 MedGPT Advice")
                with st.spinner("💡 Generating advice using MedGPT..."):
                    # Prepare prompt
                    advice_prompt = (
                        f"A patient submitted an image. The AI vision system described it as: \"{result['query']}\".\n\n"
                        "Based on this clinical description, provide a diagnosis if possible, recommended treatments or medications, "
                        "home care steps, and whether they need to see a doctor. Use clear and medically sound guidance. Do not ask follow-up questions."
                    )

                    # AI processing
                    try:
                        medgpt_response, _ = process_symptom_text(advice_prompt)
                        st.session_state["image_response"] = medgpt_response
                        st.success(medgpt_response)

                    except Exception as e:
                        st.error(f"🚫 Detection failed: {e}")

                    # Button to read aloud (if available)
                    if "image_response" in st.session_state and st.session_state["image_response"]:
                        if st.button("🔊 Read Aloud", key="read_image"):
                            if callable(globals().get("speak_text")):
                                speak_text(st.session_state["image_response"])
                            else:
                                st.warning("Speech function not available.")
        except Exception as e:
            st.error(f"🚫 Visual detection failed: {e}")

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