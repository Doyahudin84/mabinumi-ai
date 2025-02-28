import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load pre-trained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Function to generate a response from T5
def generate_response(input_text):
    # Encode the input text for the model
    inputs = tokenizer("generate: " + input_text, return_tensors="pt")
    
    # Get the model's prediction (output)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=100)
    
    # Decode the generated text and return the result
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("AI Asisten dengan T5")
st.write("Tanya saya sesuatu!")

# Initialize session state to keep track of chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Function to display chat history
def display_history():
    for message in st.session_state.history:
        st.write(f"**User**: {message['user']}")
        st.write(f"**AI**: {message['ai']}")

# User input text
user_input = st.text_input("Masukkan pertanyaan atau pernyataan:")

if user_input:
    # Generate the response using T5
    response = generate_response(user_input)

    # Save the user input and AI response in the chat history
    st.session_state.history.append({"user": user_input, "ai": response})

    # Display the updated chat history
    display_history()
