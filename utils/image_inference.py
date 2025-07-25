# image_inference.py
from PIL import Image
import os
import tempfile
from io import BytesIO
import requests
import base64
import streamlit as st  # ✅ Added to use st.secrets

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or "your-groq-api-key"

def llama_vision_fallback(image_bytes):
    print("🔁 Using LLaMA 4 Scout vision model")

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful medical assistant. Analyze visual input and provide an objective medical description of what you see. Do not speculate."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze this medical image and describe any visible abnormalities, conditions, symptoms, or visual cues relevant for medical diagnosis. Keep the response factual and concise."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Pass to MedGPT or other downstream logic here
        medgpt_input = {
            "query": content,
            "source": "llama-vision"
        }
        return "llama-vision", [medgpt_input]

    except Exception as e:
        print(f"❌ Groq LLaMA Vision API call failed: {e}")
        return "unknown", [{"message": "LLaMA fallback failed", "error": str(e)}]

def detect_skin_condition(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((640, 640))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file, format="JPEG", quality=85)
        tmp_path = tmp_file.name

    try:
        return llama_vision_fallback(image_bytes)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
