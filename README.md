# BizDecisions AI — Customer Feedback Intelligence Platform

A production-ready SaaS application for analyzing customer reviews using NLP and Machine Learning.

## Features

- 🧠 **Sentiment Analysis** — BERT-based multi-class sentiment detection (Positive / Neutral / Negative)
- 📊 **Aspect-Based Analysis** — Identifies aspects (Price, Service, Quality, etc.) from reviews
- 📈 **Real-time Dashboard** — Interactive visualizations with streaming updates
- 🔍 **Competitor Comparison** — Side-by-side brand sentiment benchmarking
- 💬 **LDA Topic Modeling** — Extracts complaint themes from negative reviews
- 🤖 **AI Diagnostics** — Groq/Gemini-powered business insight generation
- 🔐 **User Authentication** — Secure login with bcrypt
- 🗄️ **Database Persistence** — SQLite-backed review history

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| NLP Model | HuggingFace Transformers (BERT) |
| ML | scikit-learn, XGBoost |
| Database | SQLite |
| API | FastAPI |
| Containerization | Docker |

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run cus_app.py
```

### Running the API

```bash
uvicorn api:app --reload
```

### Docker

```bash
docker build -t bizdecisions-ai .
docker run -p 8501:8501 bizdecisions-ai
```

## Project Structure

```
├── cus_app.py          # Main Streamlit application
├── api.py              # FastAPI backend
├── app.py              # Core ML pipeline
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
└── reviews.csv         # Sample review dataset
```

## License

MIT License — see [LICENSE](LICENSE) for details.
