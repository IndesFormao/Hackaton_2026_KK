import os
from pathlib import Path
from collections import Counter

directory = '.'  # твоя директория

# Шаг 1: Собираем ВСЕ файлы
all_files = []
for root, dirs, files in os.walk("./ПДнDataset/share"):
    for file in files:
        full_path = os.path.join(root, file)
        all_files.append(full_path)

# Шаг 2: Сохраняем список файлов
with open('files_list.txt', 'w', encoding='utf-8') as f:
    for path in all_files:
        f.write(path + '\n')

# Шаг 3: Извлекаем расширения
extensions = [Path(path).suffix.lower() for path in all_files if Path(path).suffix]
unique_ext = sorted(set(extensions))
ext_count = Counter(extensions)

# Шаг 4: Сохраняем расширения
with open('extensions_count.txt', 'w', encoding='utf-8') as f:
    for ext, count in sorted(ext_count.items()):
        f.write(f"{ext}: {count}\n")

print(f"✅ Файлы: {len(all_files)} шт. в files_list.txt")
print(f"✅ Расширения: {len(unique_ext)} видов")
print("📁 extensions_count.txt")
