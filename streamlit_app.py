import streamlit as st
from transformers import pipeline

# Pastikan PyTorch terinstal dan mendeteksi perangkat yang digunakan
import torch
device = 0 if torch.cuda.is_available() else -1  # Gunakan GPU jika tersedia

# Inisialisasi pipeline dengan model yang lebih ringan (distilgpt2)
assistant = pipeline("text-generation", model="distilgpt2", device=device)

# Judul aplikasi
st.title("AI Asisten Pembelajaran")

# Deskripsi aplikasi
st.write("""
AI Asisten Pembelajaran:  
Halo! Saya di sini untuk membantu Anda belajar. Ketikkan pertanyaan atau topik yang ingin Anda bahas, dan saya akan memberikan jawaban berbasis AI.
""")

# Input dari pengguna
user_input = st.text_input("Tanyakan sesuatu:", "")

if user_input:
    # Generate response dari AI
    response = assistant(user_input, max_length=150, num_return_sequences=1)
    ai_reply = response[0]['generated_text']
    
    # Menampilkan jawaban dari AI
    st.write(f"**AI Asisten Pembelajaran**: {ai_reply.strip()}")
