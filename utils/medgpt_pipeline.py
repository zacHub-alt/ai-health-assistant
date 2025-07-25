import os
import pandas as pd
import ast
import difflib
from groq import Groq
import requests
import streamlit as st

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

# Load different grounding datasets
def load_grounding_data(source="usmle"):
    if source == "usmle":
        path = os.path.join("data", "MedQA-USMLE-4-options-train.csv")
        df = pd.read_csv(path)
        df = df[["question", "answer_words", "options", "correct_answer"]].dropna()
        df["options_parsed"] = df["options"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
    elif source == "afri":
        path = os.path.join("data", "afri_med_qa_15k_v2.5_phase_2_15275.csv")
        df = pd.read_csv(path)
        df = df.rename(columns={"question_clean": "question", "answer_rationale": "answer_words"})
        df["options_parsed"] = None
        df["correct_answer"] = ""
    else:
        raise ValueError("Unknown dataset selected")
    return df

# Get top-N similar examples based on text similarity
def get_contextual_examples(df, user_input, n=3):
    df = df.copy()
    df["similarity"] = df["question"].apply(
        lambda q: difflib.SequenceMatcher(None, q.lower(), user_input.lower()).ratio()
    )
    top = df.sort_values("similarity", ascending=False).head(n)
    return "\n\n".join([
        f"Example Symptom: {row['question']}\nAdvice Given: {row['answer_words']}"
        for _, row in top.iterrows()
    ])

# Use Google Places API to find nearby places
def find_nearby_places(location, keyword="pharmacy"):
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    fallback = [{"name": "Default Location", "lat": 6.5244, "lng": 3.3792}]  # Lagos fallback

    try:
        lat, lng = location["lat"], location["lng"]
        places_url = (
            f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
            f"location={lat},{lng}&radius=3000&keyword={keyword}&key={api_key}"
        )
        places_res = requests.get(places_url).json()

        places = [
            {
                "name": place["name"],
                "lat": place["geometry"]["location"]["lat"],
                "lng": place["geometry"]["location"]["lng"]
            }
            for place in places_res.get("results", [])
        ]

        return places if places else fallback

    except Exception as e:
        print(f"Place lookup error: {e}")
        return fallback

# Final model inference logic
def process_symptom_text(user_input: str, dataset="usmle") -> tuple:
    """
    Process the user symptom input and return AI-generated advice and optional medication guidance.
    Works with multiple datasets (e.g. 'usmle', 'afri') and gives safe, simple responses in a conversational tone.
    """

    grounding_df = load_grounding_data(dataset)
    few_shot = get_contextual_examples(grounding_df, user_input, n=3)

    prompt = f"""
You are a compassionate rural health assistant supporting patients in Nigeria, especially in remote areas.
Address the patient with "Dear Patient".

When someone shares their symptoms with you:

1. Start with a warm, simple tone — be caring and friendly.
2. If the symptoms strongly suggest a likely illness (e.g. malaria, flu, typhoid), say it **might be** that, clearly stating it is not a confirmed diagnosis.
3. Ask **1–2 short follow-up questions** to understand better.
4. Give **practical advice** using everyday language — for example, rest, hydration, what to eat, or home remedies that are commonly available.
5. If safe, suggest **basic over-the-counter meds** (like paracetamol or ORS), **only after** asking your follow-up questions.
6. End with a **gentle reminder** to visit a clinic or health worker if things don’t improve or get worse.

❌ Avoid technical medical terms.
❌ Never give a final diagnosis.
✅ Always aim to guide and support safely, like a trusted local health assistant.

Here are some sample cases for inspiration (use only for tone and style):
{few_shot}
Patient says:
"{user_input}"

Your response:
"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=0.5,
            max_tokens=500
        )

        message = response.choices[0].message.content.strip()
        return message, []  # <- Now always returning a tuple

    except Exception as e:
        return f"⚠️ Error generating response: {e}", []
