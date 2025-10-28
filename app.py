import streamlit as st
import threading
import time
import os

import speech_recognition as sr
import pyttsx3
from groq import Groq

# NEW: imports for PyAudio-free recording
import numpy as np
import sounddevice as sd

# === SET PAGE CONFIG FIRST ===
st.set_page_config(page_title="🗣️ AI Voice Assistant", layout="centered")

# === CONFIG ===
# SECURITY: don't hard-code keys in source; use secrets or env vars
GROQ_API_KEY = st.secrets.get("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") or os.getenv("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", "")
MODEL = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml or your environment.")

# === TTS engine setup ===
@st.cache_resource
def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', engine.getProperty('rate') - 25)
    return engine

engine = init_tts()

# === Session state for interruption and prompt ===
if "is_speaking" not in st.session_state:
    st.session_state.is_speaking = False
if "interrupt_input" not in st.session_state:
    st.session_state.interrupt_input = ""

# === Voice Input without PyAudio ===
def get_voice_input(duration_seconds: int = 5, samplerate: int = 16000):
    """
    Records audio via sounddevice (no PyAudio), wraps it in sr.AudioData,
    and recognizes via Google's API.
    """
    r = sr.Recognizer()
    st.info("🎤 Listening... Please speak clearly.")

    # Record raw PCM audio using sounddevice
    audio = sd.rec(int(duration_seconds * samplerate),
                   samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    samples = np.squeeze(audio)  # shape (N,)
    audio_data = sr.AudioData(samples.tobytes(), samplerate, 2)  # 16-bit = 2 bytes

    try:
        # Optional: r.adjust_for_ambient_noise equivalent is not used here because we are not using Microphone source.
        text = r.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {e}")
    return None

# === Query Model ===
def query_llm(prompt: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

# === Speak with monitoring for 'i' input ===
def speak_with_interrupt_check(text: str):
    st.session_state.is_speaking = True
    engine.say(text)

    def monitor_interrupt():
        while st.session_state.is_speaking:
            if st.session_state.interrupt_input.strip().lower() == 'i':
                engine.stop()
                st.session_state.is_speaking = False
                break
            time.sleep(0.2)

    monitor_thread = threading.Thread(target=monitor_interrupt, daemon=True)
    monitor_thread.start()

    engine.runAndWait()
    st.session_state.is_speaking = False

# === UI ===
st.title("🎤 Talk to LLM with Your Voice")

col1, col2 = st.columns([3, 1])

# Left: Ask Question
with col1:
    if st.button("🎙️ Start Speaking"):
        with st.spinner("Recording..."):
            prompt = get_voice_input()
            if prompt:
                st.success(f"✅ You said: {prompt}")
                with st.spinner("Getting AI response..."):
                    response = query_llm(prompt)
                    st.text_area("🤖 AI says:", response, height=150)
                    st.info("Speaking response...")

                    # Start speaking in a thread to allow interrupt
                    tts_thread = threading.Thread(target=speak_with_interrupt_check, args=(response,), daemon=True)
                    tts_thread.start()
            else:
                st.warning("❌ No valid input recognized.")

# Right: Interrupt Input Box
with col2:
    st.text_input("✋ Type 'i' here to interrupt speech:", key="interrupt_input")
    if st.session_state.interrupt_input.strip().lower() == 'i' and st.session_state.is_speaking:
        st.warning("🔇 Interrupt triggered by input!")

