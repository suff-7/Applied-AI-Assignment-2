# index_papers.py
import os
import re
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from PyPDF2 import PdfReader
from tqdm import tqdm


# === CONFIG ===
# Set this to your dataset root directory
ROOT_DIR = r"C:\Users\Sufyaan\Downloads\Applied AI-Assignment 2\Dataset"
OUTPUT_CSV = "corpus_index.csv"


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_title(title: str) -> str:
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s\-:()]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_pdf_title_and_firstpage(pdf_path: Path) -> Dict[str, str]:
    title = ""
    first_page = ""
    try:
        reader = PdfReader(str(pdf_path))
        # Try metadata title
        if reader.metadata and getattr(reader.metadata, "title", None):
            title = str(reader.metadata.title or "")
        # Fallback: first page text
        if not title and len(reader.pages) > 0:
            first_page = reader.pages[0].extract_text() or ""
            # Heuristic: first non-empty line as title candidate
            lines = [clean_text(l) for l in first_page.split("\n") if clean_text(l)]
            if lines:
                # Often the first 1-2 lines near top are titles; take the longest among first 3 lines
                title_candidates = lines[:3]
                title = max(title_candidates, key=len) if title_candidates else ""
    except Exception:
        # If parsing fails, leave fields blank and continue
        pass

    return {
        "pdf_title_raw": clean_text(title),
        "first_page_raw": clean_text(first_page),
        "normalized_title": normalize_title(title) if title else ""
    }


def is_pdf(file_path: Path) -> bool:
    return file_path.suffix.lower() == ".pdf"


def index_pdfs(root_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root_dir = root_dir.resolve()
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if not is_pdf(fpath):
                continue
            # Infer author as immediate parent directory name (root/Author/file.pdf)
            author = fpath.parent.name
            raw_filename = fpath.name

            # Compute file hash for deduplication across author folders
            try:
                sha256_hash = compute_sha256(fpath)
            except Exception:
                sha256_hash = ""

            # Extract title/first page for better identification
            meta = extract_pdf_title_and_firstpage(fpath)

            rows.append({
                "author": author,
                "file_path": str(fpath),
                "raw_filename": raw_filename,
                "sha256_hash": sha256_hash,
                "pdf_title_raw": meta.get("pdf_title_raw", ""),
                "first_page_raw": meta.get("first_page_raw", ""),
                "normalized_title": meta.get("normalized_title", "")
            })
    return rows


def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    print(f"Indexing PDFs under: {root}")
    rows = index_pdfs(root)

    df = pd.DataFrame(rows)
    # Create a canonical_title column: prefer normalized_title → cleaned filename stem
    def canonical_title(row):
        if row["normalized_title"]:
            return row["normalized_title"]
        stem = Path(row["raw_filename"]).stem
        return normalize_title(stem)

    df["canonical_title"] = df.apply(canonical_title, axis=1)

    # Basic stats
    total = len(df)
    unique_files_by_hash = df["sha256_hash"].nunique(dropna=True)
    print(f"Total PDFs indexed: {total}")
    print(f"Unique by SHA-256: {unique_files_by_hash}")

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved index to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
