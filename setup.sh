#!/bin/bash
# Deployment setup script

echo "🚀 Setting up Research Paper Reviewer Recommendation System..."

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📚 Downloading NLTK stopwords..."
python -c "import nltk; nltk.download('stopwords')"

# Check if models exist
if [ ! -f "tfidf_model.pkl" ]; then
    echo "⚠️  Model files not found."
    echo "💡 To build models:"
    echo "   1. Add your dataset to Dataset/ folder"
    echo "   2. Run: python index_papers.py"
    echo "   3. Run: python extract_text_and_profiles.py"
    echo "   4. Run: python build_tfidf_baseline.py"
    echo ""
    echo "🌐 Starting app in demo mode..."
else
    echo "✅ Model files found. Starting full system..."
fi

# Start Streamlit
echo "🎯 Launching Streamlit app..."
streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0