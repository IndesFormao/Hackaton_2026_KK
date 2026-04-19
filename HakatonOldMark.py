import os
import re
import csv
import time
import json

# ====== OCR ======
import pytesseract
from PIL import Image

# УКАЖИ СВОЙ ПУТЬ (если отличается)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ====== ПАТТЕРНЫ ======
PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\+?\d[\d\s\-()]{8,}\d",
    "inn": r"\b\d{10}\b|\b\d{12}\b",
    "snils": r"\b\d{3}-\d{3}-\d{3} \d{2}\b",
    "passport": r"\b\d{4} \d{6}\b",
    "card": r"\b\d{13,19}\b",
    "name": r"[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+( [А-ЯЁ][а-яё]+)?"
}

MAX_PDF_PAGES = 5
MAX_FILE_SIZE_MB = 100

# ====== ЛУН ======
def luhn_check(card_number):
    digits = [int(d) for d in card_number if d.isdigit()]
    checksum = 0
    parity = len(digits) % 2

    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit

    return checksum % 10 == 0

# ====== СКАН ======
def scan_directory(root):
    files = []
    for root_dir, dirs, filenames in os.walk(root):
        for file in filenames:
            files.append(os.path.join(root_dir, file))
    return files

# ====== ИЗВЛЕЧЕНИЕ ======
def extract_pdf(path):
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:MAX_PDF_PAGES]:
                t = page.extract_text()
                if t:
                    text += t
    except:
        pass
    return text

def extract_docx(path):
    text = ""
    try:
        from docx import Document
        doc = Document(path)
        for p in doc.paragraphs:
            text += p.text + "\n"
    except:
        pass
    return text

def extract_csv(path):
    try:
        import pandas as pd
        df = pd.read_csv(path, dtype=str)
        return df.to_string()
    except:
        return ""

def extract_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data)
    except:
        return ""

# OCR для TIF
def extract_tif(path):
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang="rus+eng")
        return text
    except:
        return ""

def extract_text(path):
    ext = path.lower().split(".")[-1]

    if ext == "pdf":
        return extract_pdf(path)
    elif ext == "docx":
        return extract_docx(path)
    elif ext == "csv":
        return extract_csv(path)
    elif ext == "json":
        return extract_json(path)
    elif ext in ["tif", "tiff"]:
        return extract_tif(path)

    return ""

# ====== КОНТЕКСТ ======
def has_context(text):
    keywords = [
        "паспорт", "снилс", "дата рождения",
        "адрес", "гражданин", "выдан"
    ]

    text = text.lower()
    return any(word in text for word in keywords)

# ====== ДЕТЕКЦИЯ ======
def is_pdn(text):
    strong = 0
    weak = 0

    for key, pattern in PATTERNS.items():
        matches = re.findall(pattern, text)

        if key == "card":
            matches = [m for m in matches if luhn_check(m)]

        if not matches:
            continue

        if key in ["passport", "snils", "inn", "card"]:
            strong += len(matches)

        if key in ["email", "phone", "name"]:
            weak += len(matches)

    if strong > 0:
        return True

    if weak >= 2 and has_context(text):
        return True

    return False

# ====== ВРЕМЯ ======
def format_time(timestamp):
    return time.strftime("%b %d %H:%M", time.localtime(timestamp)).lower()

# ====== СОХРАНЕНИЕ ======
def save_result(files, filename="result.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=',')

        writer.writerow(["size", "time", "name"])

        for file in files:
            try:
                writer.writerow([
                    os.path.getsize(file),
                    format_time(os.path.getmtime(file)),
                    os.path.basename(file)
                ])
            except:
                continue

    print("result.csv готов!")

# ====== PIPELINE ======
def process(root_folder):
    files = scan_directory(root_folder)
    result_files = []

    print(f"Всего файлов: {len(files)}")

    for i, file in enumerate(files):
        print(f"[{i+1}/{len(files)}] {file}")

        try:
            if os.path.getsize(file) > MAX_FILE_SIZE_MB * 1024 * 1024:
                continue
        except:
            continue

        ext = file.lower().split(".")[-1]

        if ext not in ["pdf", "docx", "csv", "json", "tif", "tiff"]:
            continue

        text = extract_text(file)

        if not text:
            continue

        if is_pdn(text):
            result_files.append(file)

    save_result(result_files)

# ====== ЗАПУСК ======
if __name__ == "__main__":
    ROOT = r"D:\Desktop\ПДнDataset\share"
    process(ROOT)