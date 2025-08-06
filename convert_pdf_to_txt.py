import fitz  # PyMuPDF
import os

# Set input/output folders
input_folder = "pdfs/"
output_folder = "data/"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(input_folder, filename)
        txt_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))

        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved: {txt_path}")