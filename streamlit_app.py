import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Function to generate text based on input prompt
def generate_text(prompt):
    # Encode the prompt for the model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Generate the output text
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit UI
st.title("AI Asisten dengan BART")
st.write("Masukkan prompt untuk menghasilkan teks!")

# Input: User provides a prompt
user_input = st.text_input("Masukkan prompt atau pertanyaan:")

if user_input:
    # Generate text based on the input prompt
    generated_text = generate_text(user_input)

    # Display the generated text as the AI response
    st.write(f"**AI**: {generated_text}")
