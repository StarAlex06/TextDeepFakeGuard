# models/ml/train_ml_models.py
import pandas as pd
import numpy as np
import pickle
import os
import sys

# ДОБАВЬТЕ ЭТИ СТРОКИ ДО ИМПОРТОВ
sys.path.append('../../')  # Добавляем корень проекта в путь
sys.path.append('../')     # Добавляем родительскую директорию

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Теперь импорт должен работать
try:
    from features.stylometric import StylometricFeatureExtractor
    print("Successfully imported StylometricFeatureExtractor")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    # Альтернативный способ импорта
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stylometric",
        "../../features/stylometric.py"
    )
    stylometric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stylometric)
    StylometricFeatureExtractor = stylometric.StylometricFeatureExtractor


class MLPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.extractor = StylometricFeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.vectorizers = {}

    def extract_features(self, texts):
        """Извлечение фичей для списка текстов"""
        stylo_features = []
        for text in texts:
            features = self.extractor.extract_all(text)
            stylo_features.append(list(features.values()))

        # Имена фичей для отладки
        self.stylo_feature_names = self.extractor.feature_names
        return np.array(stylo_features)

    def create_tfidf_features(self, texts, ngram_range=(1, 3), max_features=5000):
        """Создание TF-IDF фичей"""
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english',
            min_df=2
        )
        return vectorizer.fit_transform(texts), vectorizer

    def create_char_ngram_features(self, texts, ngram_range=(2, 5), max_features=3000):
        """Создание character n-gram фичей"""
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=2
        )
        return vectorizer.fit_transform(texts), vectorizer

    def train_models(self, X_train, y_train, X_val, y_val):
        """Обучение всех ML моделей"""

        # Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr

        # Linear SVM
        print("Training Linear SVM...")
        svm = LinearSVC(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000
        )
        svm.fit(X_train, y_train)
        self.models['svm'] = svm

        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf

        # Оценка на валидации
        print("\nValidation scores:")
        for name, model in self.models.items():
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]) if hasattr(model, 'predict_proba') else 0
            print(f"{name}: Accuracy = {accuracy:.4f}, ROC-AUC = {roc_auc:.4f}")

    def evaluate(self, X_test, y_test):
        """Оценка моделей на тестовом наборе"""
        results = {}

        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            print(f"\n{name.upper()}:")
            print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
                'predictions': y_pred,
                'probabilities': y_proba
            }

        return results

    def save_models(self, path='models/ml/saved_models'):
        """Сохранение обученных моделей"""
        os.makedirs(path, exist_ok=True)

        for name, model in self.models.items():
            with open(os.path.join(path, f'{name}.pkl'), 'wb') as f:
                pickle.dump(model, f)

        # Сохраняем векторизаторы и скейлеры если есть
        for name, vec in self.vectorizers.items():
            with open(os.path.join(path, f'{name}_vectorizer.pkl'), 'wb') as f:
                pickle.dump(vec, f)

        print(f"Models saved to {path}")


def main():
    # Загружаем данные
    print("Loading data...")
    train = pd.read_csv('../../data/processed/train.csv')
    val = pd.read_csv('../../data/processed/val.csv')
    test = pd.read_csv('../../data/processed/test.csv')

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Инициализируем пайплайн
    pipeline = MLPipeline()

    # 1. Извлекаем стилометрические фичи
    print("\nExtracting stylometric features...")
    X_train_stylo = pipeline.extract_features(train['text'].tolist())
    X_val_stylo = pipeline.extract_features(val['text'].tolist())
    X_test_stylo = pipeline.extract_features(test['text'].tolist())

    # 2. Создаем TF-IDF фичи
    print("Creating TF-IDF features...")
    X_train_tfidf, tfidf_vectorizer = pipeline.create_tfidf_features(train['text'].tolist())
    pipeline.vectorizers['tfidf'] = tfidf_vectorizer

    X_val_tfidf = tfidf_vectorizer.transform(val['text'].tolist())
    X_test_tfidf = tfidf_vectorizer.transform(test['text'].tolist())

    # 3. Создаем character n-gram фичи
    print("Creating character n-gram features...")
    X_train_char, char_vectorizer = pipeline.create_char_ngram_features(train['text'].tolist())
    pipeline.vectorizers['char'] = char_vectorizer

    X_val_char = char_vectorizer.transform(val['text'].tolist())
    X_test_char = char_vectorizer.transform(test['text'].tolist())

    # 4. Объединяем все фичи
    from scipy.sparse import hstack

    X_train = hstack([X_train_tfidf, X_train_char, X_train_stylo])
    X_val = hstack([X_val_tfidf, X_val_char, X_val_stylo])
    X_test = hstack([X_test_tfidf, X_test_char, X_test_stylo])

    y_train = train['label'].values
    y_val = val['label'].values
    y_test = test['label'].values

    print(f"\nFeature dimensions:")
    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # 5. Обучаем модели
    pipeline.train_models(X_train, y_train, X_val, y_val)

    # 6. Оцениваем на тестовом наборе
    results = pipeline.evaluate(X_test, y_test)

    # 7. Сохраняем модели
    pipeline.save_models()

    return results


if __name__ == "__main__":
    results = main()