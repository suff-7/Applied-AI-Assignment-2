
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  

AUTHOR_PROFILES_PKL = "author_profiles.pkl"
PAPER_TEXTS_PKL = "paper_texts.pkl"
CORPUS_INDEX_CSV = "corpus_index.csv"
AUTHOR_PAPERS_CSV = "author_papers.csv"

OUTPUT_TFIDF_MODEL = "tfidf_model.pkl"
OUTPUT_AUTHOR_VECTORS = "author_tfidf_vectors.pkl"
OUTPUT_AUTHOR_LIST = "author_list.pkl"

TEST_RESULTS_CSV = "tfidf_test_results.csv"


def clean_text(text: str) -> str:
    """Same cleaning as extract_text_and_profiles.py"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_and_clean_pdf(pdf_path: str) -> str:
    """Extract and clean text from a PDF for testing."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return clean_text(text)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""


def recommend_reviewers(
    query_text: str,
    tfidf_vectorizer: TfidfVectorizer,
    author_vectors: np.ndarray,
    author_names: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find top-k most similar authors for a given paper text.
    
    Returns: List of (author_name, similarity_score) tuples
    """
    
    query_vector = tfidf_vectorizer.transform([query_text])
    
    similarities = cosine_similarity(query_vector, author_vectors)[0]
    
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [(author_names[idx], similarities[idx]) for idx in top_k_indices]
    return results


def main():
    print("="*80)
    print("TF-IDF BASELINE REVIEWER RECOMMENDATION SYSTEM")
    print("="*80)
    
    print("\n[Step 1/5] Loading author profiles...")
    with open(AUTHOR_PROFILES_PKL, 'rb') as f:
        author_profiles = pickle.load(f)
    
    author_names = list(author_profiles.keys())
    author_texts = list(author_profiles.values())
    
    print(f"Loaded {len(author_names)} author profiles")
    
    # Build TF-IDF model
    print("\n[Step 2/5] Building TF-IDF vectorizer...")
    # Use unigrams and bigrams, limit features to top 5000 most important terms
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  
        min_df=2,  
        max_df=0.8  
    )
    
    # Fit and transform author profiles
    author_vectors = tfidf_vectorizer.fit_transform(author_texts)
    
    print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Author vector shape: {author_vectors.shape}")
    
    # Save model components
    print("\n[Step 3/5] Saving TF-IDF model...")
    with open(OUTPUT_TFIDF_MODEL, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    with open(OUTPUT_AUTHOR_VECTORS, 'wb') as f:
        pickle.dump(author_vectors, f)
    
    with open(OUTPUT_AUTHOR_LIST, 'wb') as f:
        pickle.dump(author_names, f)
    
    print(f"✓ Saved TF-IDF model to {OUTPUT_TFIDF_MODEL}")
    print(f"✓ Saved author vectors to {OUTPUT_AUTHOR_VECTORS}")
    print(f"✓ Saved author list to {OUTPUT_AUTHOR_LIST}")
    
    # Test with sample papers
    print("\n[Step 4/5] Testing with sample papers from dataset...")
    
    # Load corpus index to get test papers
    df_index = pd.read_csv(CORPUS_INDEX_CSV)
    df_papers = pd.read_csv(AUTHOR_PAPERS_CSV)
    
    # Select 5 diverse test papers (one from different authors)
    test_authors = ['Pinaki Chakraborty', 'Devendra K Tayal', 'Rama Murthy Garimella', 
                    'Nishchal K. Verma', 'Sowmini Devi']
    
    test_results = []
    
    for test_author in test_authors:
        # Get one paper from this author
        author_papers = df_index[df_index['author'] == test_author]
        if len(author_papers) == 0:
            continue
        
        # Pick first paper
        test_paper = author_papers.iloc[0]
        test_path = test_paper['file_path']
        test_title = test_paper['canonical_title'] or test_paper['raw_filename']
        
        print(f"\n--- Testing with paper by {test_author} ---")
        print(f"Paper: {test_title[:80]}...")
        
        # Extract and clean text
        paper_text = extract_and_clean_pdf(test_path)
        
        if not paper_text:
            print("⚠ Could not extract text from this paper")
            continue
        
        # Get recommendations
        recommendations = recommend_reviewers(
            paper_text, 
            tfidf_vectorizer, 
            author_vectors, 
            author_names, 
            top_k=10
        )
        
        # Display top 5
        print(f"Top 5 Recommended Reviewers:")
        # helper: normalize names for loose matching
        def normalize_name(n: str) -> str:
            if not n:
                return ""
            n = n.lower()
            n = re.sub(r"[^a-z0-9\s]", " ", n)
            n = re.sub(r"\s+", " ", n).strip()
            return n

        def same_author(a: str, b: str) -> bool:
            # consider authors same if normalized token sets overlap significantly
            na = normalize_name(a)
            nb = normalize_name(b)
            if not na or not nb:
                return False
            sa = set(na.split())
            sb = set(nb.split())
            inter = sa.intersection(sb)
            # allow match if >50% tokens overlap of the shorter name
            shorter = min(len(sa), len(sb))
            return (len(inter) >= max(1, int(0.5 * shorter)))

        for rank, (author, score) in enumerate(recommendations[:5], 1):
            match_indicator = "✓ MATCH!" if same_author(author, test_author) else ""
            print(f"  {rank}. {author:<35} (score: {score:.4f}) {match_indicator}")
        
        # Check if true author is in top-10 using normalized matching
        is_in_top_10 = False
        rank_position = None
        for i, (cand_author, _) in enumerate(recommendations[:10]):
            if same_author(cand_author, test_author):
                is_in_top_10 = True
                rank_position = i + 1
                break
        
        # Save result
        test_results.append({
            'test_author': test_author,
            'paper_title': test_title[:100],
            'top_1_recommendation': recommendations[0][0],
            'top_1_score': recommendations[0][1],
            'true_author_in_top_10': is_in_top_10,
            'true_author_rank': rank_position
        })
    
    # Save test results
    df_results = pd.DataFrame(test_results)
    df_results.to_csv(TEST_RESULTS_CSV, index=False)
    print(f"\n✓ Saved test results to {TEST_RESULTS_CSV}")
    
    # Summary
    print("\n[Step 5/5] Evaluation Summary")
    print("="*80)
    
    if len(test_results) > 0:
        in_top_10_count = sum([r['true_author_in_top_10'] for r in test_results])
        accuracy = in_top_10_count / len(test_results) * 100
        
        print(f"Papers tested: {len(test_results)}")
        print(f"True author found in Top-10: {in_top_10_count}/{len(test_results)} ({accuracy:.1f}%)")
        
        avg_rank = np.mean([r['true_author_rank'] for r in test_results if r['true_author_rank']])
        print(f"Average rank of true author: {avg_rank:.2f}")
    
    print("\n" + "="*80)
    print("✓ TF-IDF BASELINE COMPLETE!")
    print("="*80)
    print("\nNext step: Build BERT embeddings for semantic similarity")
    print("Or proceed directly to building the Streamlit UI")


if __name__ == "__main__":
    main()
