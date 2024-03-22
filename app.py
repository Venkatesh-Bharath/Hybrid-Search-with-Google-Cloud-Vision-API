import streamlit as st
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from google.cloud import vision
import os


# Set your Google Cloud API key as the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<Enter your API Key Here>"


# Now you can use the Google Cloud Vision API or any other Google Cloud service


# Initialize GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token
model = GPT2Model.from_pretrained("gpt2")

# Function to generate text embeddings
def generate_text_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    embeddings = np.resize(embeddings, (512,))  # Resize to match image embedding size
    return embeddings

# Function to extract image features using Google Cloud Vision API
def extract_image_features(image_bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.annotate_image({'image': image})
    # Extract features (here, we concatenate dominant colors and labels)
    dominant_colors = [[color.color.red, color.color.green, color.color.blue] for color in response.image_properties.dominant_colors.colors]
    labels = [annotation.description for annotation in response.label_annotations]
    image_embedding = np.concatenate([*dominant_colors, labels])
    return image_embedding

# Function to perform hybrid search using cosine similarity
def perform_hybrid_search(text_embedding, image_embedding, sample_data):
    query_embedding = np.concatenate([text_embedding, image_embedding])
    similarities = []
    for item_name, item_embedding in sample_data:
        similarity = np.dot(query_embedding, item_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding))
        similarities.append((similarity, item_name))
    sorted_data = sorted(similarities, key=lambda x: x[0], reverse=True)
    return [item_name for similarity, item_name in sorted_data[:3]]  # Return top 3 most similar items

# Streamlit app
def main():
    st.title("Hybrid Search with Google Cloud Vision API")

    # Input text
    st.write("Enter your query:")
    input_text = st.text_input("Query:")

    # Image upload
    st.write("Upload an image:")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Sample data (replace with your actual search data and corresponding embeddings)
    sample_data = [
        ("image1.jpg", np.random.rand(512)),  # Replace with actual image embedding
        ("text_data1.txt", np.random.rand(512)),  # Replace with actual text embedding
    ]

    # Process query and image
    if st.button("Search"):
        if input_text.strip() != '':
            text_embedding = generate_text_embeddings(input_text)
            if uploaded_image is not None:
                image_embedding = extract_image_features(uploaded_image.read())
            else:
                st.error("Please upload an image.")
                return
            similar_items = perform_hybrid_search(text_embedding, image_embedding, sample_data)
            st.write("Hybrid search results:")
            for item in similar_items:
                st.write(item)

if __name__ == "__main__":
    main()
