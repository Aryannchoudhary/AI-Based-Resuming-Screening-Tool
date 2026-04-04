import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import spacy
from sentence_transformers import SentenceTransformer, util
import re

st.set_page_config(page_title="AI Resume Screening", page_icon="📄")

# load model with caching to speed up subsequent runs
@st.cache_resource
def load_spacy_model():
    try:
        nlp_model = spacy.load("en_core_web_sm")
        st.success("✅ Full spaCy model loaded")
        return nlp_model
    except:
        st.info("📋 Using lightweight NLP (cloud-safe)")
        nlp_model = spacy.blank("en")

        ruler = nlp_model.add_pipe("entity_ruler")
        patterns = [
            {"label": "EMAIL", "pattern": [{"LIKE_EMAIL": True}]},
            {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"}}]}
        ]
        ruler.add_patterns(patterns)

        return nlp_model


@st.cache_resource
def load_bert():
    return SentenceTransformer('all-MiniLM-L6-v2')


nlp = load_spacy_model()
bert_model = load_bert()

# predefined skills database for keyword matching

SKILLS_DB = [
    "python", "java", "c++", "sql", "machine learning", "deep learning",
    "nlp", "data science", "pandas", "numpy", "tensorflow", "pytorch",
    "django", "flask", "html", "css", "javascript", "react", "node.js",
    "excel", "power bi", "tableau", "git", "docker", "aws"
]

def extract_skills(text):
    text = text.lower()
    return list(set([skill for skill in SKILLS_DB if skill in text]))

# function to extract text from PDF resumes

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# function to get BERT embeddings and calculate similarity

def get_embeddings(text):
    return bert_model.encode(text[:3000], convert_to_tensor=True)

def calculate_similarity(resume_embedding, job_embedding):
    return util.pytorch_cos_sim(resume_embedding, job_embedding).item()

# function to extract details using regex as fallback when NER fails

def regex_extract_details(text):
    details = {"Name": None, "Email": None, "Phone": None}

    email = re.search(r'\S+@\S+', text)
    phone = re.search(r'\b\d{10}\b', text)
    name = re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)

    details["Email"] = email.group() if email else None
    details["Phone"] = phone.group() if phone else None
    details["Name"] = name.group() if name else None

    return details


def extract_resume_details(text):
    doc = nlp(text)

    details = {
        "Name": None,
        "Email": None,
        "Phone": None,
        "Skills": extract_skills(text)
    }

    for ent in doc.ents:
        if ent.label_ == "EMAIL":
            details["Email"] = ent.text
        elif ent.label_ == "PHONE":
            details["Phone"] = ent.text

    # fallback for name
    if not details["Name"]:
        regex_data = regex_extract_details(text)
        details["Name"] = regex_data["Name"]

    return details

# function to calculate final ATS score combining similarity and skill match

def calculate_ats_score(resume_text, job_description, similarity):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)

    matched = set(resume_skills).intersection(set(jd_skills))
    skill_score = len(matched) / (len(jd_skills) + 1)

    final_score = (0.7 * similarity) + (0.3 * skill_score)

    return final_score, list(matched)

# function to rank resumes based on combined score with caching for performance
@st.cache_data
def rank_resumes(resumes, job_description):
    job_embedding = get_embeddings(job_description)
    ranked = []

    for resume in resumes:
        try:
            if not resume.name.endswith(".pdf"):
                continue

            text = extract_text_from_pdf(resume)
            resume_embedding = get_embeddings(text)

            similarity = calculate_similarity(resume_embedding, job_embedding)
            final_score, matched_skills = calculate_ats_score(text, job_description, similarity)

            details = extract_resume_details(text)

            ranked.append((details, final_score, matched_skills, text))

        except Exception as e:
            st.error(f"Error in {resume.name}: {e}")

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

# function to check diversity in ranked results (simple heuristic based on names)
def check_diversity(ranked):
    names = [r[0]["Name"] for r in ranked if r[0]["Name"]]

    if len(set(names)) < len(names) * 0.5:
        return "⚠️ Low diversity detected"
    return "✅ Fair ranking"

# Streamlit UI
st.title("📄 AI Resume Screening & Ranking System")

uploaded_resumes = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True)
job_description = st.text_area("Enter Job Description")

if st.button("Analyze & Rank"):
    if uploaded_resumes and job_description:
        with st.spinner("Processing..."):
            results = rank_resumes(uploaded_resumes, job_description)

            st.subheader("📊 Ranked Candidates")

            for i, (details, score, skills, text) in enumerate(results):
                st.markdown(f"## 🏆 Rank {i+1}: {details.get('Name', 'Unknown')}")

                st.progress(score)
                st.write(f"🎯 ATS Score: {round(score*100,2)}%")

                st.write(f"🧠 Skills Matched: {', '.join(skills) if skills else 'None'}")
                st.write(f"📧 Email: {details.get('Email','N/A')}")
                st.write(f"📞 Phone: {details.get('Phone','N/A')}")

                with st.expander("📄 View Resume"):
                    st.write(text[:1000])

                st.markdown("---")

            st.subheader("⚖️ Bias Check")
            st.write(check_diversity(results))

    else:
        st.error("Upload resumes and enter job description")