import gradio as gr
import os
import re
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer, util
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    text = ""

    if ext == 'pdf':
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    elif ext == 'docx':
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text.lower()
skills_db = [
    "python", "java", "c++", "javascript",
    "html", "css", "react", "node",
    "machine learning", "deep learning",
    "artificial intelligence", "nlp",
    "data analysis", "pandas", "numpy",
    "sql", "mysql", "mongodb",
    "git", "docker"
]
skill_mapping = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "nlp"
}
def extract_skills(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    words = text.split()

    found = set()

    # mapping based extraction
    for word in words:
        if word in skill_mapping:
            found.add(skill_mapping[word])

    # database matching
    for skill in skills_db:
        if skill in text:
            found.add(skill)

    return list(found)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
def get_resume_score(resume_text, job_description):
    resume_embed = embed_model.encode(resume_text, convert_to_tensor=True)
    jd_embed = embed_model.encode(job_description, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(resume_embed, jd_embed).item()
    score = round(similarity * 100)
    if score < 30:
        score += 10
    elif score < 60:
        score += 20
    else:
        score += 30

    return min(score, 100)

def analyze_multiple_resumes(files, job_description):
    results = []

    job_skills = extract_skills(job_description)

    for file in files:
        try:
            resume_text = extract_text(file)

            score = get_resume_score(resume_text, job_description)
            resume_skills = extract_skills(resume_text)

            matched = list(set(resume_skills) & set(job_skills))
            missing = list(set(job_skills) - set(resume_skills))

            results.append((
                os.path.basename(file.name),
                int(score),
                ", ".join(matched) if matched else "None",
                ", ".join(missing) if missing else "None"
            ))

        except Exception as e:
            results.append((
                os.path.basename(file.name),
                f"Error: {str(e)}",
                "",
                ""
            ))

    sorted_results = sorted(
        [r for r in results if isinstance(r[1], int)],
        key=lambda x: x[1],
        reverse=True
    ) + [r for r in results if not isinstance(r[1], int)]

    return sorted_results

with gr.Blocks(title="Bulk Resume Scorer", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    <h1 style='text-align: center; color: #2E86C1;'> Resume Analyzer & Job Matcher</h1>
    <p style='text-align: center; font-size: 16px; color: #117864;'>
        Upload resumes and match them with job descriptions using NLP & AI.
    </p>
    """)

    with gr.Row():
        resumes = gr.File(
            label="Upload Resumes",
            file_types=['.pdf', '.docx'],
            file_count="multiple"
        )

        job_desc = gr.Textbox(
            label="Job Description",
            lines=8,
            placeholder="Paste job description here..."
        )

    submit = gr.Button("Analyze Resumes", elem_id="submit-btn")

    results_output = gr.Dataframe(
        headers=["Resume", "Score", "Matched Skills", "Missing Skills"],
        datatype=["str", "str", "str", "str"],
        interactive=False
    )

    submit.click(
        fn=analyze_multiple_resumes,
        inputs=[resumes, job_desc],
        outputs=[results_output]
    )
    gr.HTML("""
    <style>
        #submit-btn button {
            background-color: #2ECC71 !important;
            color: white !important;
            font-size: 16px !important;
            padding: 10px 20px !important;
            border-radius: 10px !important;
            border: none !important;
        }
    </style>
    """)
if __name__ == "__main__":
    demo.launch()