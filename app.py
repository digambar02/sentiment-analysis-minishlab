import streamlit as st
import pickle
import numpy as np
from sklearn.decomposition import PCA
from model2vec import StaticModel

# Load saved PCA transformation
with open("pca_transform.pkl", "rb") as file:
    pca = pickle.load(file)

# Load trained Logistic Regression model
with open("logistic_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to generate embeddings
def generate_embeddings(text):
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    embeddings = model.encode([text])  # Generate embedding for single input
    return embeddings

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

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 20px;
        padding: 0.5rem 2rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        width: 100%;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
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
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #1E88E5; margin-bottom: 0;'>Sentiment Analyzer</h1>
            <p style='color: #666; font-size: 1.1em;'>Analyze text sentiment using AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    user_input = st.text_area(
        "Enter your text:",
        placeholder="Type or paste your text here...",
        height=150
    )
    
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                # Process Input
                embeddings = generate_embeddings(user_input)
                X_pca = pca.transform(embeddings)
                prediction = model.predict(X_pca)[0]
                probabilities = model.predict_proba(X_pca)[0]
                
                # Display Results
                sentiment = "Positive" if prediction == 1 else "Negative"
                confidence = max(probabilities)
                bg_color = "#E8F5E9" if prediction == 1 else "#FFEBEE"
                
                st.markdown(f"""
                    <div class='sentiment-box' style='background-color: {bg_color};'>
                        <h2 style='margin-bottom: 0.5rem;'>{sentiment}</h2>
                        <p style='font-size: 2rem; margin: 0;'>
                            {"😊" if prediction == 1 else "😔"}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence Scores
                st.markdown("### Confidence Scores")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        "Positive",
                        f"{probabilities[1]:.1%}",
                        delta="confidence" if prediction == 1 else None
                    )
                
                with col_b:
                    st.metric(
                        "Negative",
                        f"{probabilities[0]:.1%}",
                        delta="confidence" if prediction == 0 else None
                    )
                
                # Visualization
                st.progress(confidence)
                
        else:
            st.error("Please enter some text to analyze.")

# --- Footer ---
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #666;'>
        <p>Made with ❤️ using Minishlab Model, PCA, Logistic Regression and Streamlit</p>
    </div>
""", unsafe_allow_html=True)


#V2

# # ---- Streamlit UI ----
# st.set_page_config(page_title="AI Sentiment Classifier", page_icon="🤖", layout="centered")

# # Sidebar for branding and instructions
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3076/3076401.png", width=120)  # AI Icon
# st.sidebar.title("AI Sentiment Analyzer")
# st.sidebar.markdown("🔍 **Analyze the sentiment of any review using AI-powered embeddings & PCA!**")
# st.sidebar.info("⚡ Enter a review, and we'll predict whether it's **positive** or **negative** along with confidence scores.")

# # Main content
# st.markdown("<h2 style='text-align: center; color: #007bff;'>Sentiment Classification with Embeddings & PCA</h2>", unsafe_allow_html=True)

# st.write("✍️ **Enter a review below and click 'Predict Sentiment'**")

# # User input
# user_input = st.text_area("Enter your review:", "")

# if st.button("Predict Sentiment 🚀"):
#     if user_input:
#         # Generate embeddings
#         embeddings = generate_embeddings(user_input)

#         # Apply PCA transformation
#         X_pca = pca.transform(embeddings)

#         # Predict sentiment and probability
#         prediction = model.predict(X_pca)[0]
#         prob = model.predict_proba(X_pca)[0]  # Get probability scores
#         confidence = max(prob)  # Confidence score

#         # Sentiment Mapping
#         if prediction == 1:
#             sentiment = "Positive 😊"
#             color = "green"
#         else:
#             sentiment = "Negative 😡"
#             color = "red"

#         # Display Result
#         st.markdown(f"<h3 style='color: {color}; text-align: center;'>{sentiment}</h3>", unsafe_allow_html=True)

#         # Display probability
#         st.write("**Prediction Confidence:**")
#         st.progress(confidence)  # Visual confidence score
#         st.write(f"🔵 **Positive Probability:** {prob[1]:.2%}")
#         st.write(f"🔴 **Negative Probability:** {prob[0]:.2%}")

#     else:
#         st.warning("⚠️ Please enter a review to predict sentiment.")

#V1
# # Streamlit App UI
# st.title("Sentiment Classification with Embeddings & PCA")
# st.write("Enter a review and get sentiment prediction!")

# # User input
# user_input = st.text_area("Enter your review:", "")

# if st.button("Predict Sentiment"):
#     if user_input:
#         # Generate embeddings
#         embeddings = generate_embeddings(user_input)

#         # Apply PCA transformation
#         X_pca = pca.transform(embeddings)

#         # Predict sentiment
#         prediction = model.predict(X_pca)[0]
#         sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"

#         # Display result
#         st.write(f"**Predicted Sentiment:** {sentiment}")
#     else:
#         st.write("⚠️ Please enter a review to predict sentiment.")
