from PyPDF2 import PdfReader

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text

import re

skills_db = [
    "python", "java", "sql", "machine learning",
    "data analysis", "c++", "html", "css", "javascript"
]

def extract_skills(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    found_skills = set()

    for skill in skills_db:
        if skill in text:
            found_skills.add(skill)

    return list(found_skills)
