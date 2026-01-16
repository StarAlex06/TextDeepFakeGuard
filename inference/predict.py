# inference/predict.py
"""
Simple API for text classification
"""
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
import sys
import os

sys.path.append('../')
try:
    from features.stylometric import StylometricFeatureExtractor
except ImportError:
    print("Error importing StylometricFeatureExtractor")
    # Создаем простую версию на месте
    import re
    from collections import Counter

    class SimpleStylometricExtractor:
        def __init__(self):
            self.feature_names = []

        def extract_all(self, text):
            """Упрощенная версия без сложных зависимостей"""
            features = {}

            # Базовые признаки
            words = re.findall(r'\b\w+\b', text)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]

            features['char_count'] = len(text)
            features['word_count'] = len(words)
            features['sentence_count'] = len(sentences)
            features['avg_word_length'] = len(text) / max(len(words), 1)
            features['avg_sentence_length'] = len(words) / max(len(sentences), 1)

            # Лексическое разнообразие
            if words:
                word_counts = Counter([w.lower() for w in words if w.isalpha()])
                vocab_size = len(word_counts)
                total_words = len(words)
                features['type_token_ratio'] = vocab_size / total_words
            else:
                features['type_token_ratio'] = 0

            # Стилистические признаки
            punctuation = len(re.findall(r'[.,!?;:()\[\]{}"\'-]', text))
            features['punctuation_ratio'] = punctuation / max(len(text), 1)

            # Заглушки для остальных признаков
            for feat in ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio',
                        'pronoun_ratio', 'function_word_ratio', 'flesch_reading_ease',
                        'gunning_fog_index']:
                features[feat] = 0.0

            self.feature_names = list(features.keys())
            return features

    StylometricFeatureExtractor = SimpleStylometricExtractor

class TextClassifier:
    def __init__(self, models_dir='../models/ml/saved_models'):
        """Инициализация классификатора"""
        self.models_dir = models_dir
        self.extractor = StylometricFeatureExtractor()

        self.load_models()

    def load_models(self):
        """Загрузка моделей"""
        try:
            # Загружаем лучшую модель
            with open(os.path.join(self.models_dir, 'logistic_regression.pkl'), 'rb') as f:
                self.model = pickle.load(f)

            # Загружаем векторизаторы
            with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)

            with open(os.path.join(self.models_dir, 'char_vectorizer.pkl'), 'rb') as f:
                self.char_vectorizer = pickle.load(f)

            print("Models loaded successfully!")

        except Exception as e:
            print(f"Error loading models: {e}")
            self.model = None

    def extract_features(self, text):
        """Извлечение признаков для текста"""
        # TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform([text])

        # Character n-grams
        char_features = self.char_vectorizer.transform([text])

        # Stylometric features - вызываем БЕЗ параметра use_pos_features
        try:
            # Пробуем с параметром (если обновленный extractor)
            stylo_features = self.extractor.extract_all(text, use_pos_features=False)
        except TypeError:
            # Если не поддерживает параметр, вызываем без него
            stylo_features = self.extractor.extract_all(text)

        stylo_array = np.array(list(stylo_features.values())).reshape(1, -1)

        # Combine
        combined_features = hstack([
            tfidf_features,
            char_features,
            csr_matrix(stylo_array)
        ])

        return combined_features

    def predict(self, text):
        """Предсказание для одного текста"""
        if self.model is None:
            return {"error": "Model not loaded"}

        features = self.extract_features(text)

        try:
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]

            result = {
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'prediction': 'AI-generated' if prediction == 1 else 'Human-written',
                'label': int(prediction),
                'confidence_human': float(probability[0]),
                'confidence_ai': float(probability[1]),
                'confidence': float(max(probability)),
                'length': len(text)
            }

            return result

        except Exception as e:
            return {"error": str(e)}

    def predict_batch(self, texts):
        """Предсказание для списка текстов"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

def main():
    """Пример использования"""
    classifier = TextClassifier()

    # Тестовые примеры
    test_texts = [
        "The sun was setting over the mountains, casting long shadows across the valley.",
        "Based on quantitative analysis of the dataset, we observe a statistically significant correlation between the variables.",
        "Hello world! How are you doing today?",
        "Researchers at a university in Zurich, working on new methods for drug delivery, have made an unexpected breakthrough."
    ]

    print("Text Classification Demo")
    print("="*60)

    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}:")
        result = classifier.predict(text)

        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Details: Human={result['confidence_human']:.1%}, AI={result['confidence_ai']:.1%}")
            print(f"Text preview: {result['text_preview']}")

if __name__ == "__main__":
    main()