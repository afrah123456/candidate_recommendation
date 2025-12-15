# Resume Matcher Web App

AI-powered application that matches resumes to job descriptions using semantic similarity analysis.

---

## Overview

This web application helps recruiters and hiring managers quickly identify the best candidate matches by comparing multiple resumes against a job description. Using BERT embeddings and semantic similarity, it ranks candidates based on how well their experience aligns with job requirements.

---

## Features

- Upload job description as text input
- Batch upload multiple resumes (PDF format)
- Semantic similarity scoring using Sentence-BERT
- Ranked results with similarity percentages
- Resume summaries for quick review
- Clean, responsive web interface

---

## How It Works

1. **Text Embedding:** Converts job description and resumes into vector representations using Sentence-BERT (all-MiniLM-L6-v2 model)
2. **Similarity Calculation:** Computes cosine similarity between job description and each resume
3. **Ranking:** Displays results sorted by match score
4. **Summarization:** Provides key highlights from each resume

---

## Tech Stack

**Backend:** Flask (Python)  
**ML Model:** Sentence-Transformers (BERT)  
**PDF Processing:** PyMuPDF  
**Similarity Metric:** Cosine Similarity (scikit-learn)  
**Frontend:** HTML, CSS

---

## Installation

### Prerequisites
- Python 3.8+

### Setup
```bash
# Clone the repository
git clone <repo-url>
cd resume-matcher

# Install dependencies
pip install Flask sentence-transformers scikit-learn PyMuPDF
```

---

## Usage

### Start the Application
```bash
python app.py
```

Open your browser and go to: `http://127.0.0.1:5000`

### Using the App

1. **Paste job description** in the text area
2. **Upload PDF resumes** (one or multiple files)
3. **Click "Match Resumes"**
4. **View results** - ranked by similarity score with resume summaries

---

## Technical Details

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Lightweight BERT model optimized for semantic similarity
- 384-dimensional embeddings
- Fast inference time suitable for real-time matching

**Similarity Metric:** Cosine Similarity
- Measures angular distance between vectors
- Range: 0 (no match) to 1 (perfect match)
- Displayed as percentage for clarity

---

## Use Cases

- **Recruitment agencies** screening hundreds of candidates
- **HR departments** shortlisting applicants
- **Job seekers** optimizing resumes for specific roles
- **Career coaches** analyzing resume-job alignment


---

## Author

**Afrah Fathima**  


---

## License

MIT License
