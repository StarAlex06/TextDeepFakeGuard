"""
Simple API for text classification
"""
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
import sys
import os

sys.path.append('../')
from features.stylometric import StylometricFeatureExtractor


class TextClassifier:
    def __init__(self, models_dir='../models/ml/saved_models'):
        """Инициализация классификатора"""
        self.models_dir = models_dir
        self.models = {}
        self.vectorizers = {}
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

        # Stylometric features
        stylo_features = self.extractor.extract_all(text, use_pos_features=False)
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
        "Hello world! How are you doing today?"
    ]

    print("Text Classification Demo")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}:")
        result = classifier.predict(text)

        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Details: Human={result['confidence_human']:.1%}, AI={result['confidence_ai']:.1%}")


if __name__ == "__main__":
    main()