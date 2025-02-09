# Sentiment Analyzer

A streamlined web application that analyzes text sentiment using text embeddings, PCA, and Logistic Regression, built with Streamlit.

## Features

- Sentiment analysis with confidence scores
- Dark mode UI with responsive design
- Detailed sentiment probability breakdown
- Performance-optimized with model caching
- Mobile-friendly interface

## Tech Stack

- Python 3.13
- Streamlit
- MinishLab's Potion Retrieval Model
- scikit-learn (PCA, Logistic Regression)

## Quick Start

1. Clone and install dependencies:
```bash
git clone [<repository-url>](https://github.com/digambar02/sentiment-analysis-minishlab.git)
uv sync (If uv is already installed)
```

2. Run the application:
```bash
streamlit run app.py
```
## Project Structure

```
sentiment-analyzer/
├── app.py                         # Main application
├── pca_transform.pkl             # PCA model
├── logistic_regression_model.pkl # Classification model
├── minishlab.ipynb              # Training notebook
└── pyproject.toml             # Dependencies
```

## Note

Refer minishlab.ipynb notebook explaining embedding generation, PCA visualization with transformation, and fitting of logistic regression.

## Usage

1. Enter text in the input area
2. Click "Analyze Sentiment"
3. View results showing:
   - Sentiment prediction (Positive/Negative)
   - Confidence scores
   - Probability breakdown

## License

MIT License

## Acknowledgments

Built with MinishLab's Potion Retrieval Model and Streamlit framework.
