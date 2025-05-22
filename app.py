import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("distilbert_disaster_model")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert_disaster_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        return "Disaster" if pred == 1 else "Not Disaster"

# Streamlit UI
st.set_page_config(page_title="Disaster Tweet Classifier", layout="centered")
st.title("🚨 Disaster Tweet Classifier")
st.markdown("Classify if a tweet is related to a disaster using a BERT-based model.")

user_input = st.text_area("✏️ Enter a tweet below:")

if st.button("🔍 Predict"):
    if user_input.strip():
        result = predict(user_input)
        st.success(f"**Prediction:** {result}")
    else:
        st.warning("Please enter a tweet to classify.")
