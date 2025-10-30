import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fitz  
st.set_page_config(
    page_title="Reviewer Recommender System",
    page_icon="📄",
    layout="wide"
)

# === LOAD MODELS ===
@st.cache_resource
def load_models():
    """Load TF-IDF model and author data"""
    with open('tfidf_model.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('author_tfidf_vectors.pkl', 'rb') as f:
        author_vectors = pickle.load(f)
    with open('author_list.pkl', 'rb') as f:
        author_names = pickle.load(f)
    df_papers = pd.read_csv('author_papers.csv')
    return tfidf_vectorizer, author_vectors, author_names, df_papers

# === TEXT PROCESSING ===
def clean_text(text: str) -> str:
    """Clean text for TF-IDF"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF"""
    try:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        doc = fitz.open("temp.pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error: {e}")
        return ""

def is_likely_person_name(name: str) -> bool:
    """Filter out technical terms and non-person names from recommendations"""
    if not name or len(name.strip()) < 4:
        return False
    
    name_lower = name.lower().strip()
    
    technical_terms = [
        'sensor network', 'wireless sensor network', 'sensor', 'network',
        'algorithm', 'system', 'model', 'method', 'approach', 'technique',
        'framework', 'architecture', 'protocol', 'scheme', 'mechanism',
        'learning', 'machine learning', 'deep learning', 'neural network',
        'optimization', 'classification', 'clustering', 'detection',
        'recognition', 'prediction', 'analysis', 'processing', 'mining',
        'security', 'privacy', 'encryption', 'authentication',
        'communication', 'transmission', 'routing', 'scheduling',
        'performance', 'evaluation', 'assessment', 'comparison',
        'application', 'implementation', 'deployment', 'design',
        'agile processes', 'processes', 'web server logs', 'server logs',
        'extract knowledge', 'knowledge', 'logs', 'server', 'web server',
        'data mining', 'information extraction', 'text mining', 'data analysis',
        'software engineering', 'process improvement', 'quality assurance',
        'project management', 'workflow', 'methodology', 'best practices'
    ]
    
    if any(term in name_lower for term in technical_terms):
        return False
    
    single_tech_words = [
        'algorithm', 'system', 'network', 'sensor', 'wireless', 'digital',
        'computer', 'software', 'hardware', 'database', 'server', 'client',
        'protocol', 'security', 'privacy', 'encryption', 'authentication',
        'optimization', 'classification', 'clustering', 'detection', 'recognition',
        'processes', 'knowledge', 'logs', 'analysis', 'mining', 'extraction',
        'methodology', 'framework', 'approach', 'technique', 'model'
    ]
    
    if name_lower in single_tech_words:
        return False
    
    words = [w.strip('.,()[]{}') for w in name.split() if w.strip('.,()[]{}')]
    if len(words) < 2:
        return False
    
    capital_words = sum(1 for w in words if w and w[0].isupper())
    if capital_words < max(1, int(0.6 * len(words))):
        return False
    
    title_indicators = [
        'to extract', 'extract', 'using', 'based on', 'analysis of', 'study of',
        'approach to', 'method for', 'system for', 'algorithm for', 'model for',
        'framework for', 'technique for', 'evaluation of', 'assessment of',
        'comparison of', 'review of', 'survey of', 'investigation of'
    ]
    
    if any(indicator in name_lower for indicator in title_indicators):
        return False
    
    system_patterns = [
        r'\b(server|database|system|network|protocol|algorithm|framework)\b.*\b(logs|data|analysis|processing|mining|extraction)\b',
        r'\b(web|application|software|computer|digital|electronic)\b.*\b(server|system|platform|framework|tool)\b'
    ]
    
    if any(re.search(pattern, name_lower) for pattern in system_patterns):
        return False
    
    return True

def recommend_reviewers(query_text, tfidf_vectorizer, author_vectors, author_names, top_k=10):
    """Get top-k reviewer recommendations"""
    query_vector = tfidf_vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, author_vectors)[0]
    
    extended_k = min(len(author_names), top_k * 3)
    top_indices = np.argsort(similarities)[::-1][:extended_k]
    
    results = []
    rank = 1
    for idx in top_indices:
        author_name = author_names[idx]
        
        if not is_likely_person_name(author_name):
            continue
            
        results.append({
            'Rank': rank,
            'Author Name': author_name,
            'Similarity Score': f"{similarities[idx]:.4f}"
        })
        rank += 1
        
        if len(results) >= top_k:
            break
    
    return pd.DataFrame(results)

def main():
    # Title
    st.title("📄 Research Paper Reviewer Recommendation System")
    st.markdown("**AI-Powered Reviewer Matching using TF-IDF & Cosine Similarity**")
    st.markdown("---")
    
    try:
        tfidf_vectorizer, author_vectors, author_names, df_papers = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    
    st.sidebar.header("⚙️ Settings")
    top_k = st.sidebar.slider("Number of Reviewers", 5, 20, 10)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Statistics")
    st.sidebar.metric("Total Authors", len(author_names))
    st.sidebar.metric("Total Papers", len(df_papers))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 How It Works")
    st.sidebar.markdown("""
    1. Upload research paper (PDF)
    2. System extracts text
    3. TF-IDF vectorization
    4. Cosine similarity matching
    5. Top-k reviewers recommended
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Paper")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a research paper to find reviewers"
        )
        
        if uploaded_file:
            st.success(f"✓ File uploaded: {uploaded_file.name}")
            
            with st.spinner("Extracting text..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                cleaned_text = clean_text(raw_text)
            
            if cleaned_text:
                word_count = len(cleaned_text.split())
                st.info(f"📝 Extracted {word_count} words")
                
                if st.button("🔍 Find Reviewers", type="primary", use_container_width=True):
                    with st.spinner("Finding best reviewers..."):
                        recommendations_df = recommend_reviewers(
                            cleaned_text,
                            tfidf_vectorizer,
                            author_vectors,
                            author_names,
                            top_k=top_k
                        )
                    st.session_state['recommendations'] = recommendations_df
                    st.session_state['uploaded_file'] = uploaded_file.name
            else:
                st.error("❌ Could not extract text from PDF")
    
    with col2:
        st.header("🎯 Recommended Reviewers")
        
        if 'recommendations' in st.session_state:
            df = st.session_state['recommendations']
            
            st.success(f"**Top Match:** {df.iloc[0]['Author Name']} (Score: {df.iloc[0]['Similarity Score']})")
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"reviewers_{st.session_state['uploaded_file']}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        else:
            st.info("👈 Upload a research paper to get reviewer recommendations")

if __name__ == "__main__":
    main()
