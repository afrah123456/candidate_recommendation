import os
import fitz  # PyMuPDF
import docx
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Use stronger model for better matching
model = SentenceTransformer('all-MiniLM-L6-v2')


# ------------------------------
# Utility Functions
# ------------------------------

def extract_text_from_pdf(file_path):
    """Extract text from PDF files"""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_docx(file_path):
    """Extract text from DOCX files"""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_candidate_name(resume_text):
    """Extract candidate name from resume (assumes first non-empty line)"""
    lines = resume_text.strip().split('\n')
    for line in lines:
        if line.strip():
            return line.strip()
    return "Unknown"


def get_embedding(text):
    """Generate sentence embeddings for semantic matching"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return model.encode([""])[0]
    embeddings = model.encode(lines)
    return np.mean(embeddings, axis=0)


def generate_summary(resume_text, job_desc):
    """Generate a natural, professional summary explaining why candidate is a good fit"""
    try:
        # Use TF-IDF to find matching keywords
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        vectors = vectorizer.fit_transform([resume_text, job_desc])
        feature_names = vectorizer.get_feature_names_out()

        job_vec = vectors[1].toarray().flatten()
        resume_vec = vectors[0].toarray().flatten()

        # Find top matching keywords
        top_indices = np.argsort(resume_vec * job_vec)[::-1][:15]
        common_keywords = [feature_names[i] for i in top_indices
                           if resume_vec[i] > 0 and job_vec[i] > 0]

        if not common_keywords or len(common_keywords) < 2:
            return "Candidate shows general alignment with the position requirements."

        # Categorize skills more intelligently
        tech_stack = []
        experience_areas = []
        soft_skills = []

        # Enhanced categorization
        tech_keywords = {
            'react', 'node', 'python', 'java', 'javascript', 'typescript',
            'aws', 'docker', 'kubernetes', 'api', 'mongodb', 'sql', 'postgresql',
            'flask', 'django', 'express', 'angular', 'vue', 'html', 'css',
            'git', 'ci cd', 'microservices', 'rest', 'graphql', 'redis',
            'tensorflow', 'machine learning', 'data science', 'analytics',
            'cloud', 'devops', 'backend', 'frontend', 'full stack'
        }

        experience_keywords = {
            'years experience', 'senior', 'lead', 'architect', 'engineer',
            'developer', 'development', 'building', 'designed', 'implemented',
            'worked', 'projects', 'scalable', 'production', 'deployment'
        }

        soft_skill_keywords = {
            'team', 'leadership', 'collaboration', 'communication', 'agile',
            'scrum', 'problem solving', 'mentoring', 'cross functional',
            'stakeholder', 'project management'
        }

        # Categorize keywords
        for keyword in common_keywords:
            keyword_lower = keyword.lower()

            if any(tech in keyword_lower for tech in tech_keywords):
                if keyword_lower not in [t.lower() for t in tech_stack]:
                    tech_stack.append(keyword)
            elif any(exp in keyword_lower for exp in experience_keywords):
                if keyword_lower not in [e.lower() for e in experience_areas]:
                    experience_areas.append(keyword)
            elif any(soft in keyword_lower for soft in soft_skill_keywords):
                if keyword_lower not in [s.lower() for s in soft_skills]:
                    soft_skills.append(keyword)

        # Build natural summary
        summary_sentences = []

        # Technical skills sentence
        if tech_stack:
            tech_list = tech_stack[:4]  # Top 4 technical skills
            if len(tech_list) == 1:
                summary_sentences.append(f"Strong background in {tech_list[0]}")
            elif len(tech_list) == 2:
                summary_sentences.append(f"Proficient in {tech_list[0]} and {tech_list[1]}")
            elif len(tech_list) == 3:
                summary_sentences.append(f"Experienced with {tech_list[0]}, {tech_list[1]}, and {tech_list[2]}")
            else:
                summary_sentences.append(
                    f"Well-versed in {tech_list[0]}, {tech_list[1]}, {tech_list[2]}, and {tech_list[3]}")

        # Experience level sentence
        if experience_areas:
            if any('senior' in e.lower() or 'lead' in e.lower() for e in experience_areas):
                summary_sentences.append("demonstrated track record in senior technical roles")
            elif any('years' in e.lower() for e in experience_areas):
                summary_sentences.append("solid professional experience in software development")
            else:
                summary_sentences.append("proven experience building scalable applications")

        # Soft skills sentence
        if soft_skills:
            if len(soft_skills) == 1:
                summary_sentences.append(f"brings {soft_skills[0]} skills to the team")
            else:
                summary_sentences.append(f"excellent {soft_skills[0]} and collaborative work style")

        # Construct final summary
        if len(summary_sentences) == 0:
            return "Candidate demonstrates relevant qualifications for this position."
        elif len(summary_sentences) == 1:
            return summary_sentences[0].capitalize() + ". Good match for the role."
        elif len(summary_sentences) == 2:
            return summary_sentences[0].capitalize() + ", with " + summary_sentences[
                1] + ". Strong fit for the position."
        else:
            return summary_sentences[0].capitalize() + ", " + summary_sentences[1] + ", and " + summary_sentences[
                2] + ". Excellent match for the role."

    except Exception as e:
        print(f"Summary generation error: {e}")
        return "Candidate shows relevant qualifications and experience for this position."


# ------------------------------
# Flask Routes
# ------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        job_desc = request.form.get('job_description', '')

        if not job_desc.strip():
            return render_template('index1.html',
                                   results=[],
                                   error="Please enter a job description.")

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

            try:
                file.save(filepath)

                # Extract resume text based on file type
                if filename.lower().endswith('.pdf'):
                    resume_text = extract_text_from_pdf(filepath)
                elif filename.lower().endswith('.docx'):
                    resume_text = extract_text_from_docx(filepath)
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        resume_text = f.read()

                if not resume_text.strip():
                    continue

                name = extract_candidate_name(resume_text) or filename
                print(f"[DEBUG] Processing: {name[:30]}")

                resume_embedding = get_embedding(resume_text)
                similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]

                candidates.append({
                    'name': name,
                    'similarity': round(similarity, 4),
                    'summary': generate_summary(resume_text, job_desc)
                })

            except Exception as e:
                print(f"Error processing {filename}: {e}")
            finally:
                # Cleanup uploaded file
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass

        # Handle pasted resumes (separated by ---)
        pasted_resumes = [r.strip() for r in pasted_texts_raw.split('---') if r.strip()]
        for idx, pasted_text in enumerate(pasted_resumes):
            try:
                name = extract_candidate_name(pasted_text) or f"Pasted Resume {idx + 1}"
                resume_embedding = get_embedding(pasted_text)
                similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]

                candidates.append({
                    'name': name,
                    'similarity': round(similarity, 4),
                    'summary': generate_summary(pasted_text, job_desc)
                })
            except Exception as e:
                print(f"Error processing pasted resume {idx + 1}: {e}")

        # Sort by similarity and return top 10 matches
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        results = candidates[:10]

    return render_template('index1.html', results=results)


# ------------------------------
# App Entry Point
# ------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)