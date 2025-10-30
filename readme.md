## TEAM MEMBERS:
### Syed Sufyaan-SE22UECM046
### Ayaz Ahmed Ansari-SE22UARI024
### Hamza Babukhan-SE22UARI208
### Aryan Pawar-SE22UARI195

## Streamlit link
### https://applied-ai-assignment-2-zfwhl69pzr9hvqfeoi45yf.streamlit.app/
# 📄 Research Paper Reviewer Recommendation System

## Overview
AI-powered system that recommends suitable reviewers for research papers using TF-IDF vectorization and multi-source author extraction. Features a web interface for easy paper upload and reviewer recommendations.

## Key Features
- **Multi-Source Author Extraction**: PDF metadata → Groq LLM → Pattern matching
- **Web Interface**: Streamlit app with drag-and-drop PDF upload

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API (Optional)
Create `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
```

## Usage

### Run the Complete Pipeline
```bash
# 1. Index papers
python index_papers.py

# 2. Extract authors and build profiles
python extract_text_and_profiles.py

# 3. Train TF-IDF model
python build_tfidf_baseline.py

# 4. Launch web app
streamlit run app.py
```

Access the web interface at: http://localhost:8501

## Web Interface
- Upload PDF research papers
- Get top reviewer recommendations with similarity scores
- Export results to CSV
- View dataset statistics

## Files
- `index_papers.py` - PDF discovery and indexing
- `extract_text_and_profiles.py` - Author extraction with validation
- `build_tfidf_baseline.py` - TF-IDF model training
- `app.py` - Streamlit web application



