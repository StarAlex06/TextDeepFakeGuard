import nltk

print("Downloading NLTK resources...")

# Скачиваем необходимые ресурсы
resources = [
    'punkt',
    'averaged_perceptron_tagger',
    'stopwords',
    'wordnet',
    'omw-eng'
]

for resource in resources:
    print(f"Downloading {resource}...")
    try:
        nltk.download(resource, quiet=False)
        print(f"  ✓ {resource} downloaded")
    except Exception as e:
        print(f"  ✗ Error downloading {resource}: {e}")

print("\nAll resources downloaded!")