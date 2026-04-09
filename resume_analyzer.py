from PyPDF2 import PdfReader

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


# test part (temporary)
pdf_path = input("Enter PDF path: ")
content = read_pdf(pdf_path)

print("\n--- Extracted Text ---\n")
print(content)