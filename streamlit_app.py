import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Inisialisasi model GPT-Neo dari Hugging Face
@st.cache_resource  # Cache untuk menyimpan model dan tokenizer agar tidak dimuat ulang
def load_model():
    # Menggunakan GPT-Neo 1.3B
    model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    return generator

# Load model
generator = load_model()

# Streamlit Interface
st.title("GPT-Neo AI Assistant")
st.write("Masukkan teks untuk mendapatkan respon dari GPT-Neo.")

# User input untuk prompt
user_input = st.text_area("Tulis teks di sini:", "")

# Menampilkan hasil jika ada input
if user_input:
    with st.spinner("Menghasilkan teks..."):
        result = generator(user_input, max_length=100, num_return_sequences=1)
        generated_text = result[0]['generated_text']
        st.subheader("Hasil Output:")
        st.write(generated_text)

# Menambahkan penjelasan atau instruksi tambahan
st.markdown("""
    **Cara Kerja**:
    1. Masukkan prompt atau teks apapun di area input.
    2. GPT-Neo akan menghasilkan teks berdasarkan prompt tersebut.
    3. Anda bisa eksperimen dengan berbagai jenis input untuk melihat hasil yang berbeda.
""")
