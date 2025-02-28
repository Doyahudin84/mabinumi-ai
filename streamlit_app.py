import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Memuat model dan tokenizer GPT-J dari Hugging Face
model_name = "EleutherAI/gpt-j-6B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fungsi untuk menghasilkan teks menggunakan GPT-J
def generate_text(prompt, max_length=200):
    # Tokenisasi input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Menghasilkan teks dari model GPT-J
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

    # Decode hasil output menjadi teks
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# Membuat tampilan aplikasi Streamlit
st.title("GPT-J Text Generator")
st.write("Gunakan GPT-J untuk menghasilkan teks kreatif berdasarkan prompt Anda!")

# Input prompt dari pengguna
prompt = st.text_area("Masukkan prompt atau kalimat awal:", "Halo, bagaimana kabarmu hari ini?")

# Tombol untuk menghasilkan teks
if st.button("Hasilkan Teks"):
    if prompt:
        # Menampilkan hasil yang dihasilkan oleh GPT-J
        generated_text = generate_text(prompt)
        st.subheader("Teks yang Dihasilkan:")
        st.write(generated_text)
    else:
        st.warning("Masukkan prompt untuk menghasilkan teks!")
