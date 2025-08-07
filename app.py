import os
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for DOCX
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

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

def generate_summary(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    feature_names = vectorizer.get_feature_names_out()

    job_vec = vectors[1].toarray().flatten()
    resume_vec = vectors[0].toarray().flatten()

    top_indices = np.argsort(resume_vec * job_vec)[::-1][:5]
    common_keywords = [feature_names[i] for i in top_indices if resume_vec[i] > 0 and job_vec[i] > 0]

    if common_keywords:
        return f"Strong match in: {', '.join(common_keywords)}."
    else:
        return "Resume broadly aligns with job requirements."

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        job_desc = request.form['job_description']
        files = request.files.getlist('resumes')
        pasted_texts_raw = request.form.get('pasted_resumes', '')

        job_embedding = model.encode([job_desc])[0]
        candidates = []

        # Process uploaded files safely
        for file in files:
            if file.filename == '':
                continue  # skip empty uploads

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.lower().endswith('.pdf'):
                resume_text = extract_text_from_pdf(filepath)
            elif filename.lower().endswith('.docx'):
                resume_text = extract_text_from_docx(filepath)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    resume_text = f.read()

            name = extract_candidate_name(resume_text) or filename
            resume_embedding = model.encode([resume_text])[0]
            similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]

            candidates.append({
                'name': name,
                'similarity': round(similarity, 4),
                'summary': generate_summary(resume_text, job_desc)
            })

            # Remove file if exists
            if os.path.exists(filepath):
                os.remove(filepath)

        # Process pasted resumes separated by ---
        pasted_resumes = [r.strip() for r in pasted_texts_raw.split('---') if r.strip()]
        for idx, pasted_text in enumerate(pasted_resumes):
            name = extract_candidate_name(pasted_text) or f"Pasted Resume {idx + 1}"
            resume_embedding = model.encode([pasted_text])[0]
            similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]

            candidates.append({
                'name': name,
                'similarity': round(similarity, 4),
                'summary': generate_summary(pasted_text, job_desc)
            })

        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        results = candidates[:10]

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
