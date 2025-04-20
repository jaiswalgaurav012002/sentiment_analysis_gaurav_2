# ------------------- CACHE REDIRECTION (Must be FIRST) -------------------
import os

os.environ["HF_HOME"] = "F:/GUARAV 2/huggingface"
os.environ["HF_HUB_CACHE"] = "F:/GUARAV 2/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "F:/GUARAV 2/huggingface/transformers"

# ------------------- SYSTEM -------------------
import sys
import warnings
import subprocess
import json
import logging
from pathlib import Path
from typing import List
import numpy as np
import torch
import librosa
import soundfile as sf

# ✅ DEVICE AUTO-DETECT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {DEVICE}")
torch.set_float32_matmul_precision('high')

if DEVICE == "cpu":
    print("⚠️ CUDA not available. Running on CPU.")
else:
    print(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")

# ------------------- DATA SCIENCE -------------------
import scipy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ------------------- AUDIO / SPEECH -------------------
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
from pydub.utils import which
AudioSegment.converter = which("ffmpeg") or r"F:\\ffmpeg\\bin\\ffmpeg.exe"

from noisereduce import reduce_noise
import parselmouth

# ------------------- HUGGING FACE TRANSFORMERS -------------------
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    RagRetriever,
    RagTokenizer,
    RagSequenceForGeneration,
)

# ------------------- WHISPER -------------------
import whisper

# ------------------- FAISS (GPU/CPU) -------------------
import faiss

# ------------------- SILERO VAD (PATCHED) -------------------
try:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    (get_speech_timestamps, _, _, _, _) = utils
except Exception as e:
    print(f"❌ Silero VAD load failed: {e}")
    get_speech_timestamps = None

# ------------------- OPTIONAL: NLTK -------------------
try:
    import nltk
except ImportError:
    print("⚠️ NLTK not found. Some NLP steps may be limited.")

# ------------------- ENV LOADER -------------------
from dotenv import load_dotenv
load_dotenv()

# ------------------- SPA & NLP -------------------
import spacy
from num2words import num2words

# ------------------- TTS CONFIG -------------------
from bark import generate_audio  # ✅ Bark TTS entrypoint
from bark.generation import preload_models

USE_TTS = True          # 🔊 Use Bark TTS if available
USE_TTS_LITE = True     # 🔁 Use pyttsx3 if Bark fails
USE_TTS_ALT = True      # 🔁 Use gTTS if pyttsx3 also fails

# ------------------- HERMES LOADER -------------------
def load_hermes_model():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    preferred_model = "teknium/OpenHermes-2.5-Mistral-7B"  # ✅ OPEN access
    fallback_model_1 = "mistralai/Mistral-7B-Instruct-v0.3"  # 🔒 Gated
    fallback_model_2 = "NousResearch/Nous-Hermes-2-Mistral-7B"  # 🔒 Gated

    try:
        print(f"🔄 Loading tokenizer for {preferred_model}...")
        tokenizer = AutoTokenizer.from_pretrained(preferred_model)
        print(f"📦 Loading model: {preferred_model}")
        model = AutoModelForCausalLM.from_pretrained(preferred_model).to(DEVICE)
        print("✅ OpenHermes-2.5-Mistral-7B loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load {preferred_model}: {e}")
        try:
            print("🔁 Falling back to Mistral-7B-Instruct-v0.3...")
            tokenizer = AutoTokenizer.from_pretrained(fallback_model_1)
            model = AutoModelForCausalLM.from_pretrained(fallback_model_1).to(DEVICE)
            print("✅ Fallback Mistral-7B-Instruct-v0.3 loaded.")
        except Exception as e1:
            print(f"⚠️ Failed to load fallback_model_1: {e1}")
            try:
                print("🔁 Falling back to Nous-Hermes-2-Mistral-7B...")
                tokenizer = AutoTokenizer.from_pretrained(fallback_model_2)
                model = AutoModelForCausalLM.from_pretrained(fallback_model_2).to(DEVICE)
                print("✅ Fallback Nous-Hermes-2-Mistral-7B loaded.")
            except Exception as e2:
                print(f"❌ All Hermes model loading attempts failed: {e2}")
                raise RuntimeError("🚫 No Hermes-based model could be loaded. Please check access.")

    return tokenizer, model

# ------------------- RAG RETRIEVER -------------------
from datasets import load_dataset

def load_rag_components():
    print("📚 Loading wiki_dpr dataset with trust_remote_code=True...")
    try:
        rag_dataset = load_dataset("wiki_dpr", trust_remote_code=True)
        print("✅ wiki_dpr dataset loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load wiki_dpr: {e}")
        return None, None, None

    try:
        print("📦 Loading RAG components...")
        rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
        rag_retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-base",
            index_name="legacy",
            use_dummy_dataset=True
        )
        rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base").to(DEVICE)
        print("✅ RAG components ready.")
        return rag_tokenizer, rag_retriever, rag_model
    except Exception as e:
        print(f"❌ RAG model loading failed: {e}")
        return None, None, None

# ------------------- FALCON 3 - 7B LOADER -------------------
def setup_falcon_cache_on_f_drive():
    base_dir = "F:/huggingface"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "hub"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "transformers"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)

    os.environ["HF_HOME"] = base_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(base_dir, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_dir, "transformers")

    print("📁 Falcon cache configured:")
    print(f"   HF_HOME = {os.environ['HF_HOME']}")
    print(f"   HF_HUB_CACHE = {os.environ['HF_HUB_CACHE']}")
    print(f"   TRANSFORMERS_CACHE = {os.environ['TRANSFORMERS_CACHE']}")

def load_falcon3_model():
    model_id = "tiiuae/Falcon3-7B-Base"
    setup_falcon_cache_on_f_drive()

    try:
        print(f"🔄 Downloading Falcon3 Tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"📦 Downloading Falcon3 Model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
        print("✅ Falcon3-7B-Base loaded.")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Falcon3 loading failed: {e}")
        return None, None

# ------------------- ACTIVE MODELS -------------------
llm_tokenizer, llm_model = load_hermes_model()  # ✅
falcon3_tokenizer, falcon3_model = load_falcon3_model()
rag_tokenizer, rag_retriever, rag_model = load_rag_components()

ACTIVE_MODELS = {
    "hermes": {"tokenizer": llm_tokenizer, "model": llm_model},
    "falcon3": {"tokenizer": falcon3_tokenizer, "model": falcon3_model},
    "rag": {"tokenizer": rag_tokenizer, "retriever": rag_retriever, "model": rag_model},
}

# ------------------- STEP 3: Preprocessing (Speech + Text + Emotion) -------------------

import re
from transformers import (
    pipeline,
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification
)

from pyannote.audio.pipelines import SpeakerDiarization
import spacy

# ✅ Global device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Step 3 running on: {DEVICE.upper()}")

# 🔹 Speaker Diarization
diarization_pipeline = SpeakerDiarization.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HF_TOKEN")
)

# 🔹 SpaCy NER
nlp = spacy.load("en_core_web_sm")

# 🔹 Sentiment & Emotion (Text)
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", device=0 if DEVICE == "cuda" else -1)
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, device=0 if DEVICE == "cuda" else -1)

# 🔹 Emotion from Speech
feature_extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-er")
speech_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-large-superb-er").to(DEVICE)

# ------------------- AUDIO PREPROCESSING -------------------

def preprocess_speech(audio_path):
    waveform, _ = librosa.load(audio_path, sr=16000)
    speech_timestamps = get_speech_timestamps(waveform, sampling_rate=16000)

    if not speech_timestamps:
        print("⚠️ No speech detected by VAD.")
        return {"waveform": waveform, "diarization": None}

    # Extract voiced parts
    voiced_segments = np.concatenate([waveform[seg['start']:seg['end']] for seg in speech_timestamps])

    diarization = diarization_pipeline({"uri": "sample", "waveform": waveform})

    return {
        "waveform": voiced_segments,
        "diarization": diarization
    }

# ------------------- TEXT PREPROCESSING + NER -------------------

def expand_contractions(text):
    contractions = {"i'm": "i am", "you're": "you are", "it's": "it is"}
    return " ".join([contractions.get(word.lower(), word) for word in text.split()])

def preprocess_text(text):
    text = expand_contractions(text.lower())
    text = re.sub(r"\d+", lambda x: num2words(x.group()), text)
    text = re.sub(r"[^\w\s]", "", text)

    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return {
        "cleaned_text": text,
        "entities": entities
    }

# ------------------- TEXT-BASED SENTIMENT + EMOTION -------------------

def analyze_text_sentiment(text):
    sentiment = sentiment_pipeline(text)[0]["label"]
    emotion = max(emotion_pipeline(text)[0], key=lambda x: x["score"])["label"]
    return {
        "sentiment": sentiment,
        "emotion": emotion
    }

# ------------------- SPEECH-BASED EMOTION -------------------

def analyze_speech_emotion(audio_path):
    waveform, _ = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = speech_model(**inputs).logits
        emotion_index = torch.argmax(logits).item()

    emotions = ["Neutral", "Happy", "Sad", "Angry", "Fear"]
    return emotions[emotion_index]

# ------------------- EMOTION FUSION -------------------

def fuse_emotions(speech_emotion, text_emotion):
    scale = {"Happy": 1, "Neutral": 0, "Sad": -1, "Angry": -2, "Fear": -3}
    avg = (scale[speech_emotion] + scale[text_emotion]) / 2
    final = min(scale, key=lambda x: abs(scale[x] - avg))
    return final

# ------------------- STEP 4: Emotion-Aware LLM Response -------------------

import random
from transformers import AutoTokenizer

# ✅ Optional TTS Synthesizer (Coqui or fallback)
try:
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=DEVICE == "cuda")
    print("✅ Coqui TTS loaded successfully.")
except Exception as e:
    print(f"⚠️ TTS load failed: {e}")
    tts_model = None

# ------------------- TTS OPTIONS HANDLING -------------------
if USE_TTS:
    try:
        if not tts_model:
            tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=DEVICE == "cuda")
        print("✅ Coqui TTS confirmed active.")
    except Exception as e:
        print(f"❌ Coqui TTS failed: {e}")
        if USE_TTS_LITE:
            import pyttsx3
            tts_model = pyttsx3.init()
            print("🔁 Using pyttsx3 fallback.")
        elif USE_TTS_ALT:
            from gtts import gTTS
            tts_model = "gTTS"
            print("🔁 Using gTTS fallback.")
        else:
            tts_model = None
            print("⚠️ No TTS active. Only text replies will work.")
else:
    tts_model = None
    print("⛔ TTS globally disabled.")

# ------------------- Prompt Templates -------------------
EMOTION_TEMPLATES = {
    "Happy": "The user sounds cheerful and happy. Respond positively and engagingly.",
    "Neutral": "The user is calm and neutral. Respond helpfully and politely.",
    "Sad": "The user sounds sad. Be empathetic and supportive in your reply.",
    "Angry": "The user sounds angry or frustrated. Respond with calmness and offer solutions.",
    "Fear": "The user seems anxious or fearful. Reassure and guide gently."
}

# ------------------- Generator Function -------------------
def generate_emotion_aware_reply(text, fused_emotion, model_choice="hermes", use_tts=False):
    emotion_prefix = EMOTION_TEMPLATES.get(fused_emotion, "The user has expressed something.")
    prompt = f"{emotion_prefix}\n\nUser said:\n{text.strip()}\n\nYour response:"

    if model_choice == "hermes":
        tokenizer = ACTIVE_MODELS["hermes"]["tokenizer"]
        model = ACTIVE_MODELS["hermes"]["model"]
    elif model_choice == "falcon3":
        tokenizer = ACTIVE_MODELS["falcon3"]["tokenizer"]
        model = ACTIVE_MODELS["falcon3"]["model"]
    elif model_choice == "rag":
        tokenizer = ACTIVE_MODELS["rag"]["tokenizer"]
        retriever = ACTIVE_MODELS["rag"]["retriever"]
        model = ACTIVE_MODELS["rag"]["model"]
    else:
        raise ValueError("Invalid model_choice. Choose from: mistral, falcon3, rag")

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "prompt": prompt,
        "reply": reply,
        "emotion": fused_emotion
    }

    if use_tts and tts_model:
        audio_path = f"tts_reply_{random.randint(1000, 9999)}.wav"

        if USE_TTS_LITE and not isinstance(tts_model, str):
            tts_model.save_to_file(reply, audio_path)
            tts_model.runAndWait()
        elif USE_TTS_ALT and tts_model == "gTTS":
            gtts_obj = gTTS(text=reply)
            gtts_obj.save(audio_path)
        elif not USE_TTS_ALT and not USE_TTS_LITE:
            tts_model.tts_to_file(text=reply, file_path=audio_path)

        result["audio_path"] = audio_path

    return result

# ------------------- TEST HARNESS: CLI PIPELINE -------------------
if __name__ == "__main__":
    audio_file = "sample_audio.wav"  # Must be 16kHz mono
    user_input = "I’m feeling completely drained today..."

    print("\n🔄 Preprocessing input...")
    processed_text = preprocess_text(user_input)
    sentiment_result = analyze_text_sentiment(processed_text["cleaned_text"])
    speech_emotion = analyze_speech_emotion(audio_file)

    fused_emotion = fuse_emotions(speech_emotion, sentiment_result["emotion"])
    print(f"🧠 Fused Emotion: {fused_emotion}")

    print("\n🤖 Generating response using Hermes (top-ranked)...")
    reply = generate_emotion_aware_reply(
        text=processed_text["cleaned_text"],
        fused_emotion=fused_emotion,
        model_choice="hermes",
        use_tts=True
    )

    print("\n✅ Final Reply:", reply["reply"])
    if "audio_path" in reply:
        print(f"🎧 TTS Output saved to: {reply['audio_path']}")

# ------------------- GRADIO INTERFACE -------------------
import gradio as gr

def full_pipeline(audio_file, user_text, model_choice="hermes"):
    processed_text = preprocess_text(user_text)
    sentiment = analyze_text_sentiment(processed_text["cleaned_text"])
    speech_emotion = analyze_speech_emotion(audio_file.name)

    fused = fuse_emotions(speech_emotion, sentiment["emotion"])

    reply = generate_emotion_aware_reply(
        text=processed_text["cleaned_text"],
        fused_emotion=fused,
        model_choice=model_choice,
        use_tts=True
    )

    return reply["reply"], fused, reply.get("audio_path", None)

gr.Interface(
    fn=full_pipeline,
    inputs=[
        gr.Audio(type="file", label="🎤 Upload Audio (16kHz WAV)"),
        gr.Textbox(label="📝 Your Text"),
        gr.Radio(choices=["hermes", "falcon3", "rag"], value="hermes", label="💬 Choose LLM")
    ],
    outputs=[
        gr.Textbox(label="🤖 AI Reply"),
        gr.Textbox(label="🧠 Fused Emotion"),
        gr.Audio(label="🎧 TTS Output")
    ],
    title="🎙️ Emotion-Aware AI Assistant",
    live=True
).launch()
