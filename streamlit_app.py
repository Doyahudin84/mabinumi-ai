import streamlit as st
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Function to predict the masked word using BERT
def predict_masked_word(text):
    # Encode the text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get predictions from BERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the logits for the masked token position
    logits = outputs.logits
    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
    
    # Get the predicted token
    predicted_token_id = logits[0, mask_token_index, :].argmax(dim=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    
    return predicted_token

# Streamlit UI
st.title("AI Asisten dengan BERT")
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
    # Modify input text for the model to handle
    mask_text = user_input + " [MASK]"

    # Predict the masked word
    prediction = predict_masked_word(mask_text)

    # Save the user input and AI response in the chat history
    st.session_state.history.append({"user": user_input, "ai": prediction})

    # Display the updated chat history
    display_history()
