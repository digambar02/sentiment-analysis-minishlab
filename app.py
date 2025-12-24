import streamlit as st
import pickle
import numpy as np
from sklearn.decomposition import PCA
from model2vec import StaticModel

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Modern Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 2rem auto;
        max-width: 900px;
    }
    
    .header-section {
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        font-weight: 400;
    }
    
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        animation: scaleIn 0.5s ease-out;
    }
    
    .result-card.positive {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    
    .result-card.negative {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    
    .sentiment-emoji {
        font-size: 5rem;
        margin: 1rem 0;
        animation: bounceIn 0.6s ease-out;
    }
    
    .sentiment-label {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
        color: #1e293b;
    }
    
    .confidence-section {
        margin-top: 2rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes bounceIn {
        0% {
            transform: scale(0);
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            transform: scale(1);
        }
    }
    
    /* Streamlit element overrides */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    [data-testid="metric-container"] {
        background: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    with open("pca_transform.pkl", "rb") as file:
        pca = pickle.load(file)
    with open("logistic_regression_model.pkl", "rb") as file:
        model = pickle.load(file)
    return pca, model

pca, model = load_models()

# --- Generate Embeddings ---
@st.cache_data
def generate_embeddings(text):
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    embeddings = model.encode([text])
    return embeddings

# --- Main Layout ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-section">
        <h1 class="main-title">✨ Sentiment Analyzer</h1>
        <p class="subtitle">Discover the emotion behind your text with AI-powered analysis</p>
    </div>
""", unsafe_allow_html=True)

# Input Section
user_input = st.text_area(
    "Enter your text:",
    placeholder="Type or paste your text here... Try something like 'I absolutely love this product!' or 'This was disappointing.'",
    height=150,
    label_visibility="collapsed"
)

if st.button("🚀 Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("✨ Analyzing sentiment..."):
            # Process Input
            embeddings = generate_embeddings(user_input)
            X_pca = pca.transform(embeddings)
            prediction = model.predict(X_pca)[0]
            probabilities = model.predict_proba(X_pca)[0]
            
            # Display Results
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = max(probabilities)
            card_class = "positive" if prediction == 1 else "negative"
            emoji = "😊" if prediction == 1 else "😔"
            
            st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="sentiment-emoji">{emoji}</div>
                    <div class="sentiment-label">{sentiment}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence Scores
            st.markdown('<div class="confidence-section">', unsafe_allow_html=True)
            st.markdown("### 📊 Confidence Breakdown")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Positive",
                    f"{probabilities[1]:.1%}",
                    delta=f"+{probabilities[1]:.1%}" if prediction == 1 else None
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Negative",
                    f"{probabilities[0]:.1%}",
                    delta=f"+{probabilities[0]:.1%}" if prediction == 0 else None
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Overall Confidence Progress
            st.markdown("#### Overall Confidence")
            st.progress(confidence)
            st.markdown(f"<p style='text-align: center; color: #64748b; margin-top: 0.5rem;'>{confidence:.1%} confident in this prediction</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("""
    <div class="footer-text">
        <p>Powered by Minishlab Model • PCA • Logistic Regression</p>
        <p>Made with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
