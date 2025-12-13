# Resume Matcher Web App
This web app compares multiple resumes against a job description using semantic similarity via BERT embeddings. It's built with Flask, HTML/CSS, and PyMuPDF for PDF parsing.

# Features
Upload a job description (text input) and multiple resumes (PDFs)

Calculates cosine similarity between the job and each resume using Sentence-BERT

Displays similarity scores and resume summaries

Clean and responsive UI

# How It Works
Uses sentence-transformers (all-MiniLM-L6-v2) to embed text

Computes cosine similarity between job and each resume

Displays results on a web interface

# Requirements
Install the required Python packages:

bash
Copy
Edit
pip install Flask
pip install sentence-transformers
pip install scikit-learn
pip install PyMuPDF
🛠️ How to Run the App
Clone/download the project folder

Run the Flask server:

python app.py
Open your browser and go to:

http://127.0.0.1:5000

# Example Usage
Paste a job description

Upload 1 or more PDF resumes

Click Match Resumes

View ranked similarity results and summaries

👨‍💻 Author
Built by Afrah Fathima — as part of the SproutsAI Internship Assignment

