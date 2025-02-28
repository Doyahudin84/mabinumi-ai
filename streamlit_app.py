import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load GPT-Neo Model dan Tokenizer
@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# Judul dan Deskripsi aplikasi
st.title("GPT-Neo AI Assistant")
st.write("Selamat datang di GPT-Neo AI Assistant! Tulis pesan di bawah untuk mulai berbicara dengan asisten.")

# Menyimpan percakapan dalam session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Input dari Pengguna
user_input = st.text_input("Masukkan Pesan: ")

# Menambahkan pesan pengguna ke history
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Proses dan generate response dari GPT-Neo
    with st.spinner("Menghasilkan jawaban..."):
        result = generator(user_input, max_length=100, num_return_sequences=1)
        bot_response = result[0]['generated_text']

    # Menambahkan respons model ke percakapan
    st.session_state.messages.append({"role": "bot", "content": bot_response})

# Menampilkan chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Instruksi tambahan untuk pengguna
st.markdown("""
    **Cara Menggunakan**:
    1. Tulis pesan di kolom input untuk berbicara dengan asisten.
    2. Asisten akan memberikan respons berdasarkan pesan yang Anda kirimkan.
    3. Anda dapat melanjutkan percakapan tanpa kehilangan pesan sebelumnya.
""")
