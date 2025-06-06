import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Job Recommender", layout="centered")

# 🎨 Add background and button styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #2980b9, #6dd5fa, #ffffff);
        padding: 2rem;
    }
    textarea, .stTextInput > div > div {
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 10px;
        font-size: 16px;
        transition: background 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #34495e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 Job Recommendation System")
st.markdown("Paste your **resume** or enter your **skills** below:")

# Load dataset
df = pd.read_csv("AML-CA2/jobs.csv")
vectorizer = TfidfVectorizer(stop_words='english')
job_tfidf = vectorizer.fit_transform(df["JobDescription"])

# Recommendation function
def recommend_jobs(profile_text, top_n=5):
    profile_vec = vectorizer.transform([profile_text])
    similarities = cosine_similarity(profile_vec, job_tfidf).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    return df["JobTitle"].iloc[top_indices].tolist()

profile = st.text_area("Your Profile", height=150)

if st.button("Get Recommendations"):
    if profile.strip():
        results = recommend_jobs(profile)
        st.success("✅ Recommended Jobs:")
        for job in results:
            st.write(f"• {job}")
    else:
        st.warning("⚠️ Please enter your skills or resume content.")
