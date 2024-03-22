# Hybrid-Search-with-Google-Cloud-Vision-API
This project demonstrates a hybrid search approach combining text and image inputs using the Google Cloud Vision API and a pre-trained GPT-2 model. The system allows users to input text queries and upload images, then performs a search to find the most relevant items based on both the textual content and visual features extracted from the images.

Installation
To run this project locally, follow these steps:

Clone the repository:
git clone https://github.com/your-username/hybrid-search-gcp.git

Install the required Python packages:
[//]: # pip install streamlit numpy transformers google-cloud-vision


Set up Google Cloud credentials by exporting your Google Cloud API key as an environment variable:
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key--------------------"   < Enter your API key Here>

Usage
Run the Streamlit app:
streamlit run app.py
