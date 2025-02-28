import streamlit as st
from transformers import pipeline

try:
    # Inisialisasi pipeline untuk menggunakan GPT-2 atau model lain
    assistant = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()  # Menghentikan aplikasi jika model gagal dimuat

st.title("AI Asisten Pembelajaran")
st.write("""
AI Asisten Pembelajaran:  
Halo! Saya di sini untuk membantu Anda belajar. Ketikkan pertanyaan atau topik yang ingin Anda bahas, dan saya akan memberikan jawaban berbasis AI.
""")

user_input = st.text_input("Tanyakan sesuatu:", "")

if user_input:
    # Generate response dari AI
    response = assistant(user_input, max_length=150, num_return_sequences=1)
    ai_reply = response[0]['generated_text']
    st.write(f"**AI Asisten Pembelajaran**: {ai_reply.strip()}")
