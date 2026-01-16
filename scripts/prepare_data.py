# scripts/prepare_data_final.py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import re

# Папки
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
FEATURES_DIR = "../data/features"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

files = [
    "news_dataset.csv",
    "train-00000-of-00001 (1).parquet",
    "NewsDataset.csv",
    "train_ai.csv",
    "combined_ai_gen_dataset.csv",
    "CNN_DailyMail.csv",
    "balanced_ai_human_prompts.csv",
    "ai_human_content_detection_dataset.csv"
]


def clean_text(text):
    """Очистка текста"""
    if pd.isna(text):
        return ""
    # Убираем лишние пробелы, переносы строк
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_label(label_value, dataset_name):
    """Нормализация метки к 0/1"""
    if pd.isna(label_value):
        return 0

    try:
        # Если строка
        if isinstance(label_value, str):
            label_value = label_value.strip().lower()
            if label_value in ['0', 'false', 'human', 'real']:
                return 0
            elif label_value in ['1', 'true', 'ai', 'generated', 'fake']:
                return 1
            # Пробуем извлечь число
            numbers = re.findall(r'\d+', label_value)
            if numbers:
                num = int(numbers[0])
                return 1 if num > 0 else 0
            return 0

        # Если число
        label_num = float(label_value)
        if label_num == 0 or label_num == 1:
            return int(label_num)
        # Если другие числа
        return 1 if label_num > 0 else 0

    except:
        return 0


def process_file(file_path):
    fname = os.path.basename(file_path).lower()
    dataset_name = fname.replace(".csv", "").replace(".parquet", "").replace(".txt", "")

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
        else:
            df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return pd.DataFrame()

    df_out = pd.DataFrame()

    # Определяем колонки с текстом и метками
    text_col = None
    label_col = None

    # Приоритетный поиск колонок
    possible_text_cols = ['text', 'Text', 'text_content', 'abstract', 'content', 'article']
    possible_label_cols = ['label', 'Label', 'generated', 'target', 'is_ai', 'class']

    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if text_col:
        df_out['text'] = df[text_col].apply(clean_text)
        if label_col:
            df_out['raw_label'] = df[label_col]
            print(f"  Found: {text_col}, {label_col}")
        else:
            df_out['raw_label'] = 0
            print(f"  Found only text: {text_col} (assuming human)")
    else:
        print(f"  No text column found in {file_path}")
        return pd.DataFrame()

    # Нормализуем метки
    df_out['label'] = df_out['raw_label'].apply(lambda x: normalize_label(x, dataset_name))
    df_out['source'] = dataset_name

    # Удаляем пустые и короткие тексты
    df_out = df_out[df_out['text'].str.len() > 50]
    df_out = df_out.dropna(subset=['text', 'label'])

    print(f"  Processed: {len(df_out)} rows, labels: {dict(df_out['label'].value_counts())}")
    return df_out[['text', 'label', 'source']]


# Обработка всех файлов
all_dfs = []
for f in files:
    path = os.path.join(RAW_DIR, f)
    print(f"\nProcessing: {path}")
    df_proc = process_file(path)
    if len(df_proc) > 0:
        all_dfs.append(df_proc)

if not all_dfs:
    print("No data processed!")
    exit()

df_full = pd.concat(all_dfs, ignore_index=True)
print(f"\n{'=' * 60}")
print(f"Total dataset size: {len(df_full)}")
print(f"Label distribution:")
print(df_full['label'].value_counts())
print(f"Human: {(df_full['label'] == 0).sum()} ({((df_full['label'] == 0).sum() / len(df_full)) * 100:.1f}%)")
print(f"AI: {(df_full['label'] == 1).sum()} ({((df_full['label'] == 1).sum() / len(df_full)) * 100:.1f}%)")

# Балансировка до 60/40
print(f"\nBalancing classes...")
human_df = df_full[df_full['label'] == 0]
ai_df = df_full[df_full['label'] == 1]

# Ограничиваем majority class
target_ratio = 0.6  # 60% human, 40% AI
target_human = int(len(ai_df) * (target_ratio / (1 - target_ratio)))

if len(human_df) > target_human:
    human_df = human_df.sample(n=target_human, random_state=42)

df_balanced = pd.concat([human_df, ai_df], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"After balancing ({target_ratio * 100:.0f}/{100 - target_ratio * 100:.0f}):")
print(f"  Total: {len(df_balanced)}")
print(
    f"  Human: {(df_balanced['label'] == 0).sum()} ({(df_balanced['label'] == 0).sum() / len(df_balanced) * 100:.1f}%)")
print(f"  AI: {(df_balanced['label'] == 1).sum()} ({(df_balanced['label'] == 1).sum() / len(df_balanced) * 100:.1f}%)")

# Разделение на train/val/test
train, temp = train_test_split(
    df_balanced,
    test_size=0.3,
    random_state=42,
    stratify=df_balanced['label']
)
val, test = train_test_split(
    temp,
    test_size=0.5,
    random_state=42,
    stratify=temp['label']
)

# Сохраняем
train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

print(f"\nSplit sizes:")
print(f"Train: {len(train)}")
print(f"Val: {len(val)}")
print(f"Test: {len(test)}")

print(f"\nSaved to {PROCESSED_DIR}/")