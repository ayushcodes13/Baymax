# Project scaffolding script for Baymax
# Creates the directory and file structure for the RAG web app.

# Creating directories
mkdir -p src
mkdir -p research
mkdir -p templates
mkdir -p static
mkdir -p data
mkdir -p pdfs

# Creating files
touch src/__init__.py
touch src/helper.py
touch src/prompt.py
touch .env
touch setup.py
touch app.py
touch store_index.py
touch convert_pdf_to_txt.py
touch stt.py
touch tts.py
touch requirements.txt
touch requirements-research.txt
touch research/trials.ipynb

echo "Directory and files created successfully."
