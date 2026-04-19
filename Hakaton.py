import pytesseract
from PIL import Image
import os
import re
import csv
from pathlib import Path
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from collections import defaultdict

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class SmartPersonalDataDetector:
    def __init__(self):
        self.patterns = {
            'snils': re.compile(
                r'(?:СНИЛС|страховой номер|snils)[:\s]*(\d{3}[\-\s]?\d{3}[\-\s]?\d{3}[\-\s]?\d{2})|\b(\d{3}[\-\s]?\d{3}[\-\s]?\d{3}[\-\s]?\d{2})\b',
                re.IGNORECASE),
            'passport': re.compile(r'(?:паспорт|passport|серия|номер)[:\s]*(\d{4}[\-\s]?\d{6})|\b(\d{4}[\-\s]?\d{6})\b',
                                   re.IGNORECASE),
            'inn': re.compile(r'(?:ИНН|inn|налогоплательщика)[:\s]*(\d{12})|\b(\d{12})\b', re.IGNORECASE),
            'phone': re.compile(
                r'(?:тел|phone|моб|контакт)[:\s]*((?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})',
                re.IGNORECASE),
            'email': re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
            'fio': re.compile(r'(?:ФИО|Ф\.И\.О|фио|fio)[:\s]*([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)',
                              re.IGNORECASE)
        }

        self.exclude_patterns = [
            re.compile(r'\b(?:000|111|222|333|444|555|666|777|888|999)\b'),
            re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
            re.compile(r'\b\d+\.\d+\.\d+\b'),
            re.compile(r'\b(?:http|https|ftp)://'),
            re.compile(r'\b[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}\b', re.I),
        ]

        self.pd_keywords = [
            'паспорт', 'passport', 'снилс', 'snils', 'инн', 'inn',
            'телефон', 'phone', 'адрес', 'address', 'фио', 'fio',
            'персональные данные', 'personal data', 'конфиденциально'
        ]

    def validate_snils(self, snils_str):
        try:
            digits = re.sub(r'[\s\-]', '', snils_str)
            if len(digits) != 11 or digits == '0' * 11:
                return False

            control = int(digits[9:11])
            sum_val = sum(int(digits[i]) * (9 - i) for i in range(9))

            if sum_val < 100:
                calculated = sum_val
            elif sum_val == 100 or sum_val == 101:
                calculated = 0
            else:
                calculated = sum_val % 101
                if calculated == 100:
                    calculated = 0

            return calculated == control
        except:
            return False

    def validate_inn(self, inn_str):
        try:
            digits = re.sub(r'\D', '', inn_str)
            if len(digits) != 12 or digits == '0' * 12:
                return False

            coeff1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
            n1 = sum(int(digits[i]) * coeff1[i] for i in range(10)) % 11 % 10

            coeff2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
            n2 = sum(int(digits[i]) * coeff2[i] for i in range(11)) % 11 % 10

            return n1 == int(digits[10]) and n2 == int(digits[11])
        except:
            return False

    def validate_phone(self, phone_str):
        digits = re.sub(r'\D', '', phone_str)

        if len(digits) not in [11, 10]:
            return False

        test_patterns = [
            r'^[0-9]*(?:0{4,}|1{4,}|2{4,}|3{4,}|4{4,}|5{4,}|6{4,}|7{4,}|8{4,}|9{4,})[0-9]*$',
            r'^[0-9]*(?:0123456789|9876543210|1234567890)[0-9]*$'
        ]

        for pattern in test_patterns:
            if re.match(pattern, digits):
                return False

        if len(digits) == 11:
            operator_code = digits[1:4]
            valid_codes = ['900', '901', '902', '903', '904', '905', '906',
                           '908', '909', '910', '911', '912', '913', '914',
                           '915', '916', '917', '918', '919', '920', '921',
                           '922', '923', '924', '925', '926', '927', '928',
                           '929', '930', '931', '932', '933', '934', '935',
                           '936', '937', '938', '939', '950', '951', '952',
                           '953', '954', '955', '956', '957', '958', '959',
                           '960', '961', '962', '963', '964', '965', '966',
                           '967', '968', '969', '970', '971', '972', '973',
                           '974', '975', '976', '977', '978', '979', '980',
                           '981', '982', '983', '984', '985', '986', '987',
                           '988', '989', '991', '992', '993', '994', '995',
                           '996', '997', '998', '999']
            return operator_code in valid_codes

        return True

    def validate_passport(self, passport_str):
        digits = re.sub(r'\D', '', passport_str)
        if len(digits) != 10:
            return False

        series = digits[:4]
        number = digits[4:]
        if series == '0000' or number == '000000':
            return False

        return True

    def is_in_context(self, text, match):
        match_pos = text.find(match)
        if match_pos == -1:
            return False

        start = max(0, match_pos - 100)
        end = min(len(text), match_pos + len(match) + 100)
        context = text[start:end].lower()

        for keyword in self.pd_keywords:
            if keyword in context:
                return True

        return False

    def has_personal_data(self, text):
        if not text or len(text) < 10:
            return False

        text_lower = text.lower()
        has_keywords = any(keyword in text_lower for keyword in self.pd_keywords)

        found_valid_data = []

        for data_type, pattern in self.patterns.items():
            matches = pattern.finditer(text)

            for match in matches:
                value = match.group(1) or match.group(2)
                if not value:
                    continue

                is_false_positive = False
                for exclude_pattern in self.exclude_patterns:
                    if exclude_pattern.search(value):
                        is_false_positive = True
                        break

                if is_false_positive:
                    continue

                is_valid = False
                if data_type == 'snils':
                    is_valid = self.validate_snils(value)
                elif data_type == 'inn':
                    is_valid = self.validate_inn(value)
                elif data_type == 'phone':
                    is_valid = self.validate_phone(value)
                elif data_type == 'passport':
                    is_valid = self.validate_passport(value)
                elif data_type == 'email':
                    is_valid = True
                elif data_type == 'fio':
                    is_valid = has_keywords

                if not is_valid and data_type in ['snils', 'inn', 'passport', 'phone']:
                    is_valid = self.is_in_context(text, value)

                if is_valid:
                    found_valid_data.append({'type': data_type, 'value': value})

        if len(found_valid_data) >= 2:
            return True
        elif len(found_valid_data) == 1 and has_keywords:
            return True

        return False


class ImprovedFileProcessor:
    def __init__(self, max_workers=4):
        self.detector = SmartPersonalDataDetector()
        self.max_workers = max_workers
        self.cache_file = "ocr_cache_improved.json"
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_cache(self):
        try:
            if len(self.cache) > 1000:
                self.cache = dict(list(self.cache.items())[-1000:])
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f)
        except:
            pass

    def get_file_hash(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read(65536)).hexdigest()
        except:
            return None

    def find_files(self, directory):
        directory_path = Path(directory)
        extensions = {'.pdf', '.docx', '.xlsx', '.xls',
                      '.tif', '.tiff', '.jpg', '.jpeg', '.png'}

        for file_path in directory_path.rglob('*'):
            if file_path.suffix.lower() in extensions:
                yield file_path

    def extract_text_from_image(self, filepath, lang='rus+eng'):
        try:
            file_hash = self.get_file_hash(filepath)
            if file_hash and file_hash in self.cache:
                return self.cache[file_hash]

            with Image.open(filepath) as img:
                if img.mode != 'L':
                    img = img.convert('L')

                if img.width < 1500:
                    ratio = 1500 / img.width
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                configs = ['--oem 3 --psm 6', '--oem 3 --psm 3']

                texts = []
                for config in configs:
                    text = pytesseract.image_to_string(img, lang=lang, config=config)
                    texts.append(text)

                best_text = max(texts, key=lambda t: sum(c.isdigit() for c in t))
                best_text = re.sub(r'\s+', ' ', best_text).strip()

                if file_hash and len(best_text) > 50:
                    self.cache[file_hash] = best_text

                return best_text

        except:
            return ""

    def extract_text_from_pdf(self, filepath):
        try:
            import fitz
            doc = fitz.open(filepath)
            text_parts = []

            for page_num in range(min(len(doc), 100)):
                page = doc[page_num]
                page_text = page.get_text()

                if len(page_text.strip()) < 100:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = self.extract_text_from_image(img)
                    if ocr_text:
                        text_parts.append(ocr_text)
                else:
                    text_parts.append(page_text)

                combined = ' '.join(text_parts)
                if len(combined) > 20000:
                    break

            doc.close()
            return ' '.join(text_parts)

        except:
            return ""

    def extract_text_from_docx(self, filepath):
        try:
            from docx import Document
            doc = Document(filepath)
            text_parts = []

            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)

            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_parts.append(' | '.join(row_text))

            return ' '.join(text_parts)

        except:
            return ""

    def extract_text_from_excel(self, filepath):
        try:
            import pandas as pd
            text_parts = []

            excel_file = pd.ExcelFile(filepath)
            for sheet_name in excel_file.sheet_names[:10]:
                df = pd.read_excel(filepath, sheet_name=sheet_name, nrows=1000)

                for col in df.columns:
                    col_data = df[col].astype(str)
                    non_empty = col_data[col_data != 'nan']
                    if not non_empty.empty:
                        text_parts.append(f"{sheet_name} - {col}: {' '.join(non_empty.head(100))}")

            return ' '.join(text_parts)

        except:
            return ""

    def process_file(self, filepath):
        try:
            file_path_obj = Path(filepath)
            file_ext = file_path_obj.suffix.lower()

            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(filepath)
            elif file_ext == '.docx':
                text = self.extract_text_from_docx(filepath)
            elif file_ext in ['.xlsx', '.xls']:
                text = self.extract_text_from_excel(filepath)
            elif file_ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
                text = self.extract_text_from_image(filepath)
            else:
                return None

            if text and len(text) > 20:
                if self.detector.has_personal_data(text):
                    size = os.path.getsize(filepath)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))

                    return {
                        'size': self.format_size(size),
                        'time': mod_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'name': file_path_obj.name,
                        'full_path': str(filepath),
                        'extension': file_ext
                    }

            return None

        except:
            return None

    def format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def process_directory(self, input_dir, output_csv):
        files = list(self.find_files(input_dir))
        if not files:
            return

        total = len(files)
        print(f"Found {total} files, processing with {self.max_workers} workers...")

        results = []
        processed = 0
        found_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_file, str(f)): f for f in files}

            for future in futures:
                processed += 1
                result = future.result()

                if result:
                    results.append(result)
                    found_count += 1

                if processed % 10 == 0 or processed == total:
                    print(f"Progress: {processed}/{total} files ({found_count} with PD found)")

        self.save_cache()

        if results:
            with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['size', 'time', 'name', 'full_path', 'extension'])
                writer.writeheader()
                writer.writerows(results)

        print(f"Done. Processed: {processed}, PD found: {found_count}, Results saved to: {output_csv}")


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "result_pd.csv"

    processor = ImprovedFileProcessor(max_workers=4)
    processor.process_directory(input_dir, output_csv)