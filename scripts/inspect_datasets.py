import pandas as pd

files = [
    "C:/Users/alexe/OneDrive/Рабочий стол/deepfake/TextDeepFakeGuard/data/raw/news_dataset.csv",
    "C:/Users/alexe/OneDrive/Рабочий стол/deepfake/TextDeepFakeGuard/data/raw/train-00000-of-00001 (1).parquet",
    "C:/Users/alexe/OneDrive/Рабочий стол/deepfake/TextDeepFakeGuard/data/raw/NewsDataset.csv",
    "C:/Users/alexe/OneDrive/Рабочий стол/deepfake/TextDeepFakeGuard/data/raw/train_ai.csv",
    "C:/Users/alexe/OneDrive/Рабочий стол/deepfake/TextDeepFakeGuard/data/raw/combined_ai_gen_dataset.csv",
    "C:/Users/alexe/OneDrive/Рабочий стол/deepfake/TextDeepFakeGuard/data/raw/CNN_DailyMail.csv",
    "C:/Users/alexe/OneDrive/Рабочий стол/deepfake/TextDeepFakeGuard/data/raw/balanced_ai_human_prompts.csv",
    "C:/Users/alexe/OneDrive/Рабочий стол/deepfake/TextDeepFakeGuard/data/raw/ai_human_content_detection_dataset.csv",
]

for f in files:
    print("=" * 50)
    print(f)

    if f.endswith(".csv"):
        df = pd.read_csv(f)
    else:
        df = pd.read_parquet(f)

    print(df.columns)
    print(df.head(10))
    print(len(df))
