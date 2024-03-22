# Hybrid Search with Google Cloud Vision API and GPT-2

This project demonstrates a hybrid search approach combining text and image inputs using the Google Cloud Vision API and a pre-trained GPT-2 model. The system allows users to input text queries and upload images, then performs a search to find the most relevant items based on both the textual content and visual features extracted from the images.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
git clone https://github.com/your-username/hybrid-search-gcp.git


2. Install the required Python packages:
```bash
pip install streamlit numpy transformers google-cloud-vision
```
3.Set up Google Cloud credentials by exporting your Google Cloud API key as an environment variable:
Set your Google Cloud API key as the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Enter your key Here"

4.Usage
Run the Streamlit app:
```bash
streamlit run app.py
```
