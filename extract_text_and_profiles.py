import re
import pickle
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import requests
import json
import os
import time

import pandas as pd
import fitz
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import hashlib

from dotenv import load_dotenv
load_dotenv()
try:
    import requests
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if GROQ_API_KEY:
        print("⚡ Using Groq API for author extraction")
        print("✓ Groq API key configured")
        groq_available = True
        groq_headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        groq_model = "llama-3.1-8b-instant"
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
    else:
        print("⚠️ No Groq API key found")
        print("💡 To use Groq:")
        print("   1. Get API key from https://console.groq.com")
        print("   2. Add to .env file: GROQ_API_KEY=your_key_here")
        print("🔄 Using pattern-based fallback for now")
        groq_available = False
        groq_headers = {}
        groq_model = None
        groq_url = None
        
except Exception as e:
    print(f"⚠️ Groq configuration failed: {e}")
    print("💡 Using pattern-based fallback")
    groq_available = False
    groq_headers = {}
    groq_model = None
    groq_url = None

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

CORPUS_INDEX_CSV = "corpus_index.csv"
OUTPUT_TEXTS_PKL = "paper_texts.pkl"
OUTPUT_AUTHOR_PROFILES_PKL = "author_profiles.pkl"
OUTPUT_AUTHOR_PAPERS_CSV = "author_papers.csv"

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
        return ""

def extract_text_with_fallback(pdf_path: str) -> tuple[str, dict]:
    try:
        doc = fitz.open(pdf_path)
        
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        
        metadata = doc.metadata or {}
        
        cleaned_metadata = {}
        for key, value in metadata.items():
            if value and isinstance(value, str):
                cleaned_metadata[key.lower()] = value.strip()
        
        doc.close()
        
        return text.strip(), cleaned_metadata
        
    except Exception as e:
        print(f"Error extracting text and metadata from {pdf_path}: {e}")
        return "", {}

def extract_abstract_and_keywords(text: str) -> str:
    abstract_match = re.search(r'abstract[:\s]+(.*?)(?:introduction|keywords|1\.|i\.)', 
                              text, re.IGNORECASE | re.DOTALL)
    keywords_match = re.search(r'keywords?[:\s]+(.*?)(?:\n\n|introduction|1\.)', 
                              text, re.IGNORECASE | re.DOTALL)
    weighted_text = text
    if abstract_match:
        abstract = abstract_match.group(1)
        weighted_text = abstract + " " + abstract + " " + text
    if keywords_match:
        keywords = keywords_match.group(1)
        weighted_text = keywords + " " + keywords + " " + keywords + " " + weighted_text
    return weighted_text

def extract_firstpage_text(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = doc[0].get_text() if doc.page_count > 0 else ""
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting first page from {pdf_path}: {e}")
        return ""

def is_valid_author(name):
    blacklist = [
        "Department", "India", "University", "Science", "Institute", "Technology",
        "School", "Center", "Laboratory", "Abstract", "Introduction", "View",
        "Copyright", "Notes", "Data", "Results", "Discussion", "Figures", "Tables",
        "Email", "Corresponding", "Author", "Professor", "Researcher",
        "Digital Object", "Object Identifier", "Market Performance", "Understanding",
        "Kashmere Gate", "Article History", "Senior Member", "Fellow Member",
        "Research Scholar", "Graduate Student", "Assistant Professor", "Associate Professor",
        "Computer Applications", "Computer Science", "Information Technology", "Data Science",
        "Machine Learning", "Artificial Intelligence", "Software Engineering", "Computer Engineering",
        "SEE PROFILE", "Senior Member", "Article history", "i n f o", "PROFILE",
        "Member IEEE", "Fellow IEEE", "IEEE", "ACM", "Published", "Received",
        "Accepted", "Available online", "Keywords", "References", "Conclusion",
        "Acknowledgment", "Bibliography", "Index Terms", "Vol", "Issue", "Page",
        "DOI", "ISSN", "ISBN", "Proceedings", "Conference", "Journal", "Transaction",
        "Review", "Editor", "Board", "Committee", "Society", "Association",
        "International", "National", "Research", "College", "Faculty", "Staff",
        "Graduate", "Student", "PhD", "Masters", "Bachelor", "Degree",
        "Vishwavidyalaya", "Universidad", "Universidade", "Université", "Universität",
        "Campus", "Foundation", "Corporation", "Company", "Limited", "Ltd", "Inc",
        "Group", "Team", "Division", "Section", "Unit", "Branch", "Office",
        "Uttar Pradesh", "Tamil Nadu", "Andhra Pradesh", "West Bengal", "Maharashtra",
        "Karnataka", "Kerala", "Gujarat", "Rajasthan", "Punjab", "Haryana",
        "BITS Pilani", "Pilani", "IIT", "NIT", "IIIT", "MIT", "Stanford", "Harvard",
        "Cambridge", "Hyderabad Campus", "Delhi Campus", "Goa Campus", "BITS", 
        "Campus", "College", "Academy", "Polytechnic", "Technical", "Engineering",
        "Cellulose", "Nanocrystalline", "Polymer", "Chemical", "Material", "Compound",
        "induces", "regulated", "dependent", "based", "using", "applied", "proposed",
        "analysis", "study", "approach", "method", "algorithm", "model", "system",
        "title:", "Title", "Paper", "Article", "Study", "Analysis", "Review",
        "Survey", "Investigation", "Evaluation", "Assessment", "Comparison",
        "Fruit Disease Detection", "Power PDF Create", "Windows User", "Wireless Sensor Network",
        "Energy Ef", "Codebook Transfer", "Disease Detection", "PDF Create", "Sensor Network",
        "Detection", "Transfer", "Network", "User", "Create", "Power", "Energy", "Wireless",
        "PDF", "Windows", "Fruit", "Disease", "Sensor", "Codebook", "Object", "Digital",
        "Fourier Transform", "Transform", "Fourier", "Fast Fourier", "Discrete Fourier",
        "Adobe", "Microsoft", "Google", "Apple", "Oracle", "IBM", "Intel", "Nvidia",
        "Software", "Application", "Program", "Tool", "Platform", "Framework",
        "Operating System", "Database", "Server", "Client", "Browser", "Plugin",
        "Algorithm", "Protocol", "Interface", "Architecture", "Infrastructure", "Framework",
        "Methodology", "Technique", "Approach", "Solution", "Implementation", "Deployment",
        "Performance", "Optimization", "Enhancement", "Improvement", "Development",
        "Experiment", "Simulation", "Modeling", "Validation", "Verification", "Testing",
        "Measurement", "Calculation", "Computation", "Processing", "Analysis", "Synthesis"
    ]
    
    name = name.strip()
    if not name or len(name) < 4:
        return False
    
    if any(kw.lower() in name.lower() for kw in blacklist):
        return False
    
    if sum(1 for c in name if c.isalpha()) < len(name) * 0.6:
        return False
    
    words = [w.strip('.,()[]{}') for w in name.split() if w.strip('.,()[]{}')]
    if len(words) < 2 or len(words) > 4:
        return False
    
    for word in words:
        if not word or len(word) < 1:
            continue
        alpha_chars = sum(1 for c in word if c.isalpha())
        if alpha_chars < len(word) * 0.7:
            return False
    
    if any(char.isdigit() for char in name):
        return False
        
    if len(name) > 50:
        return False
    
    if len(name) > 4 and name.isupper():
        return False
        
    if ':' in name or ';' in name:
        return False
    
    non_name_indicators = [
        'using', 'based', 'applied', 'proposed', 'improved', 'enhanced', 'novel',
        'new', 'fast', 'efficient', 'optimal', 'advanced', 'modern', 'intelligent',
        'automatic', 'automated', 'smart', 'adaptive', 'robust', 'secure', 'safe',
        'high', 'low', 'large', 'small', 'multi', 'single', 'double', 'triple',
        'first', 'second', 'third', 'final', 'initial', 'preliminary', 'comprehensive'
    ]
    
    name_lower = name.lower()
    if any(indicator in name_lower for indicator in non_name_indicators):
        return False
    
    for word in words:
        if len(word) < 2 or not word[0].isupper():
            return False
        clean_word = word.replace('-', '').replace("'", '')
        if not clean_word.isalpha():
            return False
    
    if len(words) < 2 or len(words) > 4:
        return False
    
    if len(words) >= 2:
        technical_suffixes = ['tion', 'sion', 'ment', 'ness', 'ity', 'ism', 'ogy', 'ics', 'al', 'ic', 'ous', 'ive']
        technical_words = 0
        for word in words:
            word_lower = word.lower()
            if any(word_lower.endswith(suffix) for suffix in technical_suffixes):
                technical_words += 1
        
        if technical_words >= len(words) - 1:
            return False
    
    name_exact = name.strip()
    problematic_exact_matches = [
        "Fruit Disease Detection", "Power PDF Create", "Windows User", "Wireless Sensor Network",
        "Energy Ef", "Codebook Transfer", "Disease Detection", "PDF Create", "Sensor Network",
        "Dilip Kumar Chakrabarti"
    ]
    
    if name_exact in problematic_exact_matches:
        return False
    
    common_tech_words = {"detection", "transfer", "network", "user", "create", "power", "energy", 
                        "wireless", "pdf", "windows", "fruit", "disease", "sensor", "codebook",
                        "system", "algorithm", "method", "approach", "analysis", "study", "fourier",
                        "transform", "discrete", "fast", "continuous", "wavelet", "signal", "processing"}
    
    name_words_lower = {word.lower() for word in words}
    if len(name_words_lower & common_tech_words) >= len(words):
        return False
    
    return True

def _fix_ocr_noise(s: str) -> str:
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r'(?<=\b[ A-Za-z])[0](?=[A-Za-z])', 'o', s)
    return s

def find_title_index(lines_list):
    candidates = []
    for i, l in enumerate(lines_list[:12]):
        if len(l) < 10:
            continue
        low = l.lower()
        metadata_indicators = ['abstract', 'introduction', 'keywords', '©', 'doi', 'www', 'http',
                             'digital object identifier', 'access', 'received', 'accepted', 'published',
                             'corresponding author', 'email', '@', 'university', 'department', 'institute']
        if any(k in low for k in metadata_indicators):
            continue
        alpha_ratio = sum(1 for c in l if c.isalpha()) / len(l)
        if alpha_ratio < 0.6:
            continue
        candidates.append((len(l), i, l))
    if not candidates:
        return 0
    candidates.sort(reverse=True)
    return candidates[0][1]

def parse_author_names(text: str, metadata: dict = None) -> list[str]:
    authors = []
    
    if metadata:
        metadata_authors = extract_authors_from_metadata(metadata)
        if metadata_authors:
            print(f"📋 Found {len(metadata_authors)} authors from PDF metadata")
            return metadata_authors
    
    if groq_available:
        try:
            groq_authors = extract_authors_groq(text)
            if groq_authors:
                print(f"🤖 Groq extracted {len(groq_authors)} authors")
                return groq_authors
        except Exception as e:
            print(f"⚠️ Groq extraction failed: {e}")
    
    pattern_authors = extract_authors_pattern_fallback(text)
    if pattern_authors:
        print(f"🔍 Pattern matching found {len(pattern_authors)} authors")
        return pattern_authors
    
    print("❌ No authors extracted by any method")
    return []

def extract_authors_from_metadata(metadata: dict) -> list[str]:
    authors = []
    
    author_fields = ['author', 'authors', 'creator', 'subject']
    
    for field in author_fields:
        if field in metadata:
            author_text = metadata[field]
            
            if author_text:
                potential_authors = re.split(r'[,;]|and\b|\band\b', author_text)
                
                for author in potential_authors:
                    author = author.strip()
                    if author and is_valid_author(author):
                        author = re.sub(r'\([^)]*\)', '', author)
                        author = re.sub(r'\s+', ' ', author).strip()
                        
                        if len(author) > 2 and author not in authors:
                            authors.append(author)
    
    return authors[:10]

def compute_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error computing MD5 for {file_path}: {e}")
        return ""

def extract_title(text: str, filename: str = "", metadata: dict = None) -> str:
    if metadata:
        title_fields = ['title', 'subject', 'dc:title']
        for field in title_fields:
            if field in metadata and metadata[field]:
                title = metadata[field].strip()
                if len(title) > 5:
                    return title
    
    lines = text.split('\n')
    valid_lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 5]
    
    if valid_lines:
        title_idx = find_title_index(valid_lines)
        if title_idx < len(valid_lines):
            return valid_lines[title_idx]
    
    if filename:
        title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        return title
    
    return "Unknown Title"

def extract_authors_groq(text):
    if not groq_available:
        return extract_authors_pattern_fallback(text)
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    
    header_text = ' '.join(lines[:15])
    
    header_text = _fix_ocr_noise(header_text)
    
    if len(header_text) > 800:
        header_text = header_text[:800]
    
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert at extracting author names from academic papers. Extract only the author names from the given text, one per line. Do not include titles, affiliations, or any other information."
            },
            {
                "role": "user", 
                "content": f"Extract the author names from this academic paper header:\n\n{header_text}\n\nReturn only the names, one per line:"
            }
        ]

        response = requests.post(
            groq_url,
            headers=groq_headers,
            json={
                "model": groq_model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 200,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Groq API returned status {response.status_code}: {response.text[:200]}")
        
        result = response.json()
        response_text = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        if not hasattr(extract_authors_groq, '_first_call_done'):
            print(f"⚡ Sample Groq response: {response_text[:100]}...")
            extract_authors_groq._first_call_done = True
        
        valid_authors = []
        
        if response_text:
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'^[-*•]\s*', '', line)
                line = re.sub(r'(?i)^(dr\.?|prof\.?|mr\.?|ms\.?|mrs\.?)\s*', '', line)
                line = re.sub(r'(?i)\s*(phd|ph\.d|m\.d|md|jr\.?|sr\.?)$', '', line)
                line = line.strip(' .,')
                
                if not line or len(line) < 4:
                    continue
                
                words = line.split()
                if len(words) < 2 or len(words) > 4:
                    continue
                
                valid_name = True
                for word in words:
                    if len(word) < 2 or not word[0].isupper():
                        valid_name = False
                        break
                    if not word.replace('-', '').replace("'", '').isalpha():
                        valid_name = False
                        break
                
                if not valid_name:
                    continue
                
                common_non_names = [
                    'Computer', 'Applications', 'Science', 'Technology', 'Engineering',
                    'Information', 'Systems', 'Management', 'Business', 'Analysis',
                    'Performance', 'Research', 'Development', 'Design', 'Digital',
                    'Advanced', 'International', 'National', 'Modern', 'University',
                    'Institute', 'Department', 'School', 'College', 'Abstract',
                    'Introduction', 'Keywords', 'Corresponding', 'Author', 'Email'
                ]
                
                if any(word in common_non_names for word in words):
                    continue
                
                if is_valid_author(line):
                    valid_authors.append(line)
        
        valid_authors = list(dict.fromkeys(valid_authors))
        
        if valid_authors:
            return valid_authors[:5]
            
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "rate limit" in error_str.lower():
            print("⚠️ Groq rate limit reached, waiting and using fallback...")
            time.sleep(1)
        else:
            print(f"⚠️ Groq extraction failed: {error_str[:80]}...")
        return extract_authors_pattern_fallback(text)
    
    return extract_authors_pattern_fallback(text)

def extract_authors_pattern_fallback(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    head = lines[:15]
    
    header_text = ' '.join(head)
    header_text = _fix_ocr_noise(header_text)
    
    names = []
    
    for line in head[:8]:
        if len(line) > 60 or any(word in line.lower() for word in ['abstract', 'introduction', 'keywords', 'university', 'department']):
            continue
            
        author_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\b'
        matches = re.findall(author_pattern, line)
        
        for match in matches:
            if is_valid_author(match):
                names.append(match)
    
    author_line_pattern = r'([A-Z][a-z]+(?: [A-Z][a-z]*){1,3}(?:\s*,\s*[A-Z][a-z]+(?: [A-Z][a-z]*){1,3})*)'
    
    for line in head[:10]:
        if len(line) < 100:
            matches = re.findall(author_line_pattern, line)
            for match in matches:
                potential_authors = [name.strip() for name in match.split(',')]
                for author in potential_authors:
                    if is_valid_author(author):
                        names.append(author)
    
    return list(dict.fromkeys(names))

def main():
    print("Loading corpus index...")
    df = pd.read_csv(CORPUS_INDEX_CSV)
    print(f"Loaded {len(df)} papers from {df['author'].nunique()} folder authors")

    print("\n[Step 1/3] Extracting text from PDFs...")
    paper_texts: Dict[str, str] = {}
    unique_papers = df.drop_duplicates(subset=['sha256_hash'])
    sha_to_text = {}

    for idx, row in tqdm(unique_papers.iterrows(), total=len(unique_papers), desc="Extracting text"):
        sha_hash = row['sha256_hash']
        file_path = row['file_path']

        raw_text = extract_text_from_pdf(file_path)
        weighted_text = extract_abstract_and_keywords(raw_text)
        cleaned_text = clean_text(weighted_text)
        if cleaned_text:
            sha_to_text[sha_hash] = cleaned_text
    print(f"Extracted text from {len(sha_to_text)} unique papers")

    with open(OUTPUT_TEXTS_PKL, 'wb') as f:
        pickle.dump(sha_to_text, f)
    print(f"Saved paper texts to {OUTPUT_TEXTS_PKL}")

    print("\n[Step 2/3] Building true author profiles (from actual co-authors)...")
    author_profiles: Dict[str, List[str]] = defaultdict(list)
    author_paper_mapping: Dict[str, List[str]] = defaultdict(list)

    for idx, row in tqdm(unique_papers.iterrows(), total=len(unique_papers), desc="Extracting authors"):
        sha_hash = row['sha256_hash']
        file_path = row['file_path']
        cleaned_text = sha_to_text.get(sha_hash, "")

        firstpage_text, metadata = extract_text_with_fallback(file_path)
        actual_authors = parse_author_names(firstpage_text, metadata)

        if actual_authors and cleaned_text:
            for author in actual_authors:
                author_profiles[author].append(cleaned_text)
                title = row['canonical_title'] if 'canonical_title' in row else row['raw_filename']
                author_paper_mapping[author].append(title)

    raw_profiles = {author: ' '.join(texts) for author, texts in author_profiles.items()}
    def clean_author_label(name: str) -> str:
        if not name:
            return ""
        
        name = re.sub(r'(?i)\bet al\b|\bsee discussions?\b|\bprofile\b|\bview\b', '', name)
        name = re.sub(r'(?i)digital\s*object\s*identifier.*|doi.*|accessed.*|access.*|online.*', '', name)
        name = re.sub(r'(?i)www\.|http|\.com|\.org|\.edu|@|\./|\.\./', '', name)
        name = re.sub(r'(?i)^(dr\.?|prof\.?|mr\.?|ms\.?|mrs\.?)?\s*', '', name)
        name = re.sub(r'(?i)\s*(phd|ph\.d|m\.d|md|jr\.?|sr\.?)$', '', name)
        
        name = re.sub(r'(?i)\b(bits\s*pilani|hyderabad\s*campus|delhi\s*campus|goa\s*campus)\b', '', name)
        name = re.sub(r'(?i)\b(university|institute|college|department|school|center)\b.*', '', name)
        
        name = re.sub(r"[^A-Za-z0-9\.\-\' ]+", ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        words = name.split()
        if len(words) >= 2:
            cleaned_words = [words[0]]
            for i in range(1, len(words)):
                if words[i].lower() != words[i-1].lower():
                    cleaned_words.append(words[i])
            name = ' '.join(cleaned_words)
        
        return name

    cleaned_profiles = {}
    cleaned_mapping = {}
    for author, text_blob in raw_profiles.items():
        cleaned = clean_author_label(author)
        if not cleaned:
            continue
        if not is_valid_author(cleaned):
            continue
        
        technical_combinations = [
            'Computer Applications', 'Computer Science', 'Information Technology',
            'Data Science', 'Machine Learning', 'Artificial Intelligence',
            'Software Engineering', 'Computer Engineering', 'Information Systems',
            'Digital Marketing', 'Business Analytics', 'Market Research',
            'Performance Analysis', 'System Design', 'Network Security'
        ]
        
        if any(tech.lower() == cleaned.lower() for tech in technical_combinations):
            continue
        if cleaned in cleaned_profiles:
            cleaned_profiles[cleaned] += ' ' + text_blob
            cleaned_mapping[cleaned].append(author)
        else:
            cleaned_profiles[cleaned] = text_blob
            cleaned_mapping[cleaned] = [author]

    final_author_profiles = cleaned_profiles
    print(f"Built true profiles for {len(final_author_profiles)} extracted authors (raw: {len(raw_profiles)})")

    with open(OUTPUT_AUTHOR_PROFILES_PKL, 'wb') as f:
        pickle.dump(final_author_profiles, f)
    print(f"Saved author profiles to {OUTPUT_AUTHOR_PROFILES_PKL}")

    print("\n[Step 3/3] Saving author-paper mapping (true authors)...")
    mapping_rows = []
    for author, papers in author_paper_mapping.items():
        for paper_title in papers:
            mapping_rows.append({
                'author': author,
                'paper_title': paper_title,
                'paper_count': len(papers)
            })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(OUTPUT_AUTHOR_PAPERS_CSV, index=False)
    print(f"Saved author-paper mapping to {OUTPUT_AUTHOR_PAPERS_CSV}")

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total unique papers: {len(sha_to_text)}")
    print(f"Total extracted authors: {len(final_author_profiles)}")
    if len(final_author_profiles) > 0:
        print(f"Average papers per author: {len(mapping_df) / len(final_author_profiles):.2f}")

    if len(mapping_df) > 0:
        top_counts = mapping_df['author'].value_counts().head()
        print("\nTop 5 authors by extracted paper count:")
        for author, count in top_counts.items():
            print(f"  {author}: {count} papers")

    print("\n✓ Text extraction and actual author profile creation complete!")
    print("Next step: TF-IDF vectorization")

if __name__ == "__main__":
    main()
