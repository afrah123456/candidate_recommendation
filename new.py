import os
import fitz  # PyMuPDF
import docx
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import openai

# Optional: load OpenAI API key from env
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Use stronger model
model = SentenceTransformer('all-mpnet-base-v2')

# ------------------------------
# Utility Functions
# ------------------------------

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_candidate_name(resume_text):
    lines = resume_text.strip().split('\n')
    for line in lines:
        if line.strip():
            return line.strip()
    return "Unknown"

def get_embedding(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return model.encode([""])[0]
    embeddings = model.encode(lines)
    return np.mean(embeddings, axis=0)

# Enhanced TF-IDF summary
def generate_summary(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    feature_names = vectorizer.get_feature_names_out()

    job_vec = vectors[1].toarray().flatten()
    resume_vec = vectors[0].toarray().flatten()

    top_indices = np.argsort(resume_vec * job_vec)[::-1][:8]
    common_keywords = [feature_names[i] for i in top_indices if resume_vec[i] > 0 and job_vec[i] > 0]

    if common_keywords:
        summary = f"Strong match based on skills like {', '.join(common_keywords[:-1])} and {common_keywords[-1]}."
    else:
        summary = "Resume aligns broadly with the job requirements but no strong keyword overlaps detected."

    return summary

# Optional: LLM-powered summary (fallback to TF-IDF)
def generate_summary_with_llm(resume_text, job_desc):
    prompt = (
        "You are an AI assistant helping a recruiter evaluate candidates.\n"
        "Given a resume and a job description, summarize why this candidate may be a strong fit for the role.\n\n"
        f"Job Description:\n{job_desc}\n\n"
        f"Resume:\n{resume_text}\n\n"
        "Summary:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("LLM summary failed:", e)
        return generate_summary(resume_text, job_desc)

# ------------------------------
# Flask Routes
# ------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        job_desc = request.form['job_description']
        files = request.files.getlist('resumes')
        pasted_texts_raw = request.form.get('pasted_resumes', '')

        job_embedding = get_embedding(job_desc)
        candidates = []

        # Handle uploaded files
        for file in files:
            if file.filename == '':
                continue

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract resume text
            if filename.lower().endswith('.pdf'):
                resume_text = extract_text_from_pdf(filepath)
            elif filename.lower().endswith('.docx'):
                resume_text = extract_text_from_docx(filepath)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    resume_text = f.read()

            name = extract_candidate_name(resume_text) or filename
            print(f"[DEBUG] Extracted from {name[:20]}: {resume_text[:300]}")

            resume_embedding = get_embedding(resume_text)
            similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]

            candidates.append({
                'name': name,
                'similarity': round(similarity, 4),
                'summary': generate_summary_with_llm(resume_text, job_desc)
            })

            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)

        # Handle pasted resumes
        pasted_resumes = [r.strip() for r in pasted_texts_raw.split('---') if r.strip()]
        for idx, pasted_text in enumerate(pasted_resumes):
            name = extract_candidate_name(pasted_text) or f"Pasted Resume {idx + 1}"
            resume_embedding = get_embedding(pasted_text)
            similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]

            candidates.append({
                'name': name,
                'similarity': round(similarity, 4),
                'summary': generate_summary_with_llm(pasted_text, job_desc)
            })

        # Sort and limit top matches
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        results = candidates[:10]

    return render_template('index1.html', results=results)

# ------------------------------
# App Entry
# ------------------------------

if __name__ == '__main__':
    app.run(debug=True)
