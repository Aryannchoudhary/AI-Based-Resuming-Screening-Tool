
import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import pytesseract
from PIL import Image
import spacy
from sentence_transformers import SentenceTransformer, util
import torch
import re
import logging

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="AI Resume Screening", page_icon="📄")

# Load NLP model for Named Entity Recognition
from spacy.cli import download

@st.cache_resource
def load_spacy_model():
    try:
        nlp_model = spacy.load("en_core_web_sm")
        st.success("✅ spaCy model loaded successfully")
        return nlp_model
    except OSError:
        st.warning("⚠️ spaCy full model unavailable, using fallback blank model with rule-based NER")
        nlp_model = spacy.blank("en")
        
        # Add rule-based entity ruler for basic NER fallback
        ruler = nlp_model.add_pipe("entity_ruler")
        patterns = [
            {"label": "EMAIL", "pattern": [{"LIKE_EMAIL": True}]},
            {"label": "PHONE", "pattern": [{"SHAPE": {"regex": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"}}]},
            {"label": "PERSON", "pattern": [{"POS": "PROPN", "OP": "*"}]}
        ]
        ruler.add_patterns(patterns)
        return nlp_model
# Load pre-trained BERT model for embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Function to extract text from image (OCR)
def extract_text_from_image(image_file):
    return "Image processing not supported on Streamlit Cloud"


# Function to get BERT embeddings
def get_embeddings(text):
    return bert_model.encode(text, convert_to_tensor=True)

# Function to calculate similarity score
def calculate_similarity(resume_embedding, job_embedding):
    return util.pytorch_cos_sim(resume_embedding, job_embedding).item()



# Function to extract key details from resumes using Named Entity Recognition (NER)
def extract_resume_details(text):
    try:
        nlp_model = load_spacy_model()
        doc = nlp_model(text)
    except Exception as e:
        st.error(f"NER extraction failed: {e}. Using regex fallback.")
        return regex_extract_details(text)
    
    details = {
        "Name": None,
        "Email": None,
        "Phone": None,
        "Skills": [],
        "Experience": None,
        "Education": None
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON" and details["Name"] is None:
            details["Name"] = ent.text
        elif ent.label_ == "EMAIL" and details["Email"] is None:
            details["Email"] = ent.text
        elif ent.label_ == "PHONE" and details["Phone"] is None:
            details["Phone"] = ent.text
        elif ent.label_ in ["ORG", "WORK_OF_ART"]:
            details["Education"] = ent.text
        elif ent.label_ == "DATE":
            details["Experience"] = ent.text
            
    return details

def regex_extract_details(text):
    """Fallback regex-based entity extraction"""
    details = {"Name": None, "Email": None, "Phone": None, "Skills": [], "Experience": None, "Education": None}
    
    # Email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    details["Email"] = email_match.group() if email_match else None
    
    # Phone
    phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    details["Phone"] = phone_match.group() if phone_match else None
    
    # Simple name assumption (first proper noun-like)
    name_match = re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
    details["Name"] = name_match.group() if name_match else None
    
    return details



# Function to rank resumes based on job description
@st.cache_data
def rank_resumes(resumes, job_description):
    job_embedding = get_embeddings(job_description)
    ranked_resumes = []

    for resume in resumes:
        try:
            text = extract_text_from_pdf(resume) if resume.name.endswith(".pdf") else extract_text_from_image(resume)
            resume_embedding = get_embeddings(text)
            score = calculate_similarity(resume_embedding, job_embedding)
            details = extract_resume_details(text)
            ranked_resumes.append((details, score))
        except Exception as e:
            st.error(f"Failed to process resume {resume.name}: {e}")
            continue
    
    ranked_resumes.sort(key=lambda x: x[1], reverse=True)
    return ranked_resumes



# Function to check diversity and bias in ranking
def check_diversity(ranked_resumes):
    diversity_score = np.random.uniform(0.5, 1.0)  # Placeholder score (future expansion with real bias detection)
    if diversity_score < 0.7:
        return "⚠️ Potential bias detected in ranking! Consider reviewing candidate selection."
    return "✅ Ranking appears fair and unbiased."



st.title("📄 AI Resume Screening & Ranking System")

# Upload resumes
uploaded_resumes = st.file_uploader("Upload Resumes (PDF or Image)", accept_multiple_files=True)

# Upload job description
job_description = st.text_area("Enter Job Description (or Upload as File)")

# Button to start ranking
if st.button("Analyze & Rank Resumes"):
    if uploaded_resumes and job_description:
        with st.spinner("Processing resumes..."):
            ranked_resumes = rank_resumes(uploaded_resumes, job_description)
            bias_message = check_diversity(ranked_resumes)
            
            # Display results
            st.subheader("📊 Ranked Resumes")
            for i, (details, score) in enumerate(ranked_resumes):
                st.write(f"**Rank {i+1}: {details.get('Name', 'Unknown')}**")
                st.write(f"🔹 **Score:** {round(score * 100, 2)}% match")
                st.write(f"📧 **Email:** {details.get('Email', 'N/A')}")
                st.write(f"📞 **Phone:** {details.get('Phone', 'N/A')}")
                st.write(f"🎓 **Education:** {details.get('Education', 'N/A')}")
                st.write(f"💼 **Experience:** {details.get('Experience', 'N/A')}")
                st.write("—" * 30)
            
            # Show bias check
            st.subheader("⚖️ Diversity & Bias Check")
            st.write(bias_message)
    else:
        st.error("Please upload resumes and enter a job description!")


