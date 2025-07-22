from langdetect import detect
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Load medGPT (local or HuggingFace model)
tokenizer_medgpt = AutoTokenizer.from_pretrained("microsoft/biogpt")
model_medgpt = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
med_pipe = pipeline("text-generation", model=model_medgpt, tokenizer=tokenizer_medgpt)

# Load NLLB translation model
translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-600M")
translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "en"

def translate_to_english(text, source_lang_code):
    try:
        translation_tokenizer.src_lang = source_lang_code
        encoded = translation_tokenizer(text, return_tensors="pt")
        generated_tokens = translation_model.generate(**encoded, forced_bos_token_id=translation_tokenizer.get_lang_id("eng"))
        translated = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated
    except:
        return text

def translate_from_english(text, target_lang_code):
    try:
        translation_tokenizer.src_lang = "eng"
        encoded = translation_tokenizer(text, return_tensors="pt")
        generated_tokens = translation_model.generate(**encoded, forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang_code))
        translated = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated
    except:
        return text

def get_medical_response(user_input):
    output = med_pipe(user_input, max_length=80, do_sample=True)[0]['generated_text']
    return output

def process_symptom_text(user_input):
    detected_lang = detect_language(user_input)
    print(f"Detected language: {detected_lang}")

    if detected_lang != "en":
        user_input_en = translate_to_english(user_input, detected_lang)
    else:
        user_input_en = user_input

    med_response_en = get_medical_response(user_input_en)

    if detected_lang != "en":
        final_response = translate_from_english(med_response_en, detected_lang)
    else:
        final_response = med_response_en

    return final_response