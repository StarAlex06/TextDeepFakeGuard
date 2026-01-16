"""
Test trained ML models on test set and generate performance reports
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.sparse import hstack, csr_matrix

# Добавляем пути для импорта
sys.path.append('../../')
from features.stylometric import StylometricFeatureExtractor

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
import seaborn as sns


class ModelTester:
    def __init__(self, models_dir='saved_models'):
        """Инициализация тестера моделей"""
        self.models_dir = models_dir
        self.models = {}
        self.vectorizers = {}
        self.extractor = StylometricFeatureExtractor()

        # Загружаем модели
        self.load_models()

    def load_models(self):
        """Загрузка всех обученных моделей"""
        print("Loading models and vectorizers...")

        model_files = {
            'logistic_regression': 'logistic_regression.pkl',
            'svm': 'svm.pkl',
            'random_forest': 'random_forest.pkl'
        }

        vectorizer_files = {
            'tfidf': 'tfidf_vectorizer.pkl',
            'char': 'char_vectorizer.pkl'
        }

        # Загружаем модели
        for name, filename in model_files.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"  ✓ Loaded {name}")
            else:
                print(f"  ✗ {filename} not found")

        # Загружаем векторизаторы
        for name, filename in vectorizer_files.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.vectorizers[name] = pickle.load(f)
                print(f"  ✓ Loaded {name} vectorizer")
            else:
                print(f"  ✗ {filename} vectorizer not found")

    def extract_features(self, texts):
        """Извлечение признаков для текстов"""
        # TF-IDF features
        tfidf_features = self.vectorizers['tfidf'].transform(texts)

        # Character n-gram features
        char_features = self.vectorizers['char'].transform(texts)

        # Stylometric features (без параметра use_pos_features)
        stylo_features = []
        for text in texts:
            try:
                # Пробуем с параметром
                features = self.extractor.extract_all(text, use_pos_features=False)
            except TypeError:
                # Если не поддерживает параметр, вызываем без него
                features = self.extractor.extract_all(text)
            stylo_features.append(list(features.values()))

        stylo_array = np.array(stylo_features)

        # Объединяем все признаки
        combined_features = hstack([
            tfidf_features,
            char_features,
            csr_matrix(stylo_array)
        ])

        return combined_features

    def predict_single_text(self, text):
        """Предсказание для одного текста всеми моделями"""
        features = self.extract_features([text])

        results = {}
        for name, model in self.models.items():
            try:
                prediction = model.predict(features)[0]

                # Пробуем получить вероятности если есть
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(features)[0]
                    confidence_human = float(probability[0])
                    confidence_ai = float(probability[1])
                    confidence = max(probability)
                else:
                    # Для моделей без вероятностей (например, SVM)
                    confidence_human = None
                    confidence_ai = None
                    confidence = None

                results[name] = {
                    'prediction': 'AI-generated' if prediction == 1 else 'Human-written',
                    'label': int(prediction),
                    'confidence_human': confidence_human,
                    'confidence_ai': confidence_ai,
                    'confidence': confidence
                }
            except Exception as e:
                results[name] = {'error': str(e)}

        return results

    def evaluate_on_test_set(self, test_data_path='../../data/processed/test.csv'):
        """Оценка моделей на тестовом наборе"""
        print("\n" + "=" * 60)
        print("EVALUATION ON TEST SET")
        print("=" * 60)

        # Загружаем тестовые данные
        test_data = pd.read_csv(test_data_path)
        print(f"Test set size: {len(test_data)}")
        print(f"Class distribution:\n{test_data['label'].value_counts()}")

        # Извлекаем признаки
        print("\nExtracting features for test set...")
        X_test = self.extract_features(test_data['text'].tolist())
        y_test = test_data['label'].values

        # Оценка каждой модели
        results = {}
        for name, model in self.models.items():
            print(f"\n{'=' * 40}")
            print(f"Model: {name.upper()}")
            print('=' * 40)

            # Предсказания
            y_pred = model.predict(X_test)

            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # ROC-AUC если есть вероятности
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            except:
                y_proba = None
                roc_auc = None

            # Матрица ошибок
            cm = confusion_matrix(y_test, y_pred)

            # Детальный отчет
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            if roc_auc:
                print(f"ROC-AUC: {roc_auc:.4f}")

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

            # Сохраняем результаты
            results[name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc) if roc_auc else None,
                'confusion_matrix': cm.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_proba.tolist() if y_proba is not None else None
            }

        return test_data, results

    def save_results(self, results, output_dir='../../results/ml_results'):
        """Сохранение результатов в файлы"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Сохраняем метрики в JSON
        metrics_file = os.path.join(output_dir, f'metrics_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            # Конвертируем numpy типы в Python типы
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj

            json.dump(convert_to_serializable(results), f, indent=2)

        print(f"\nMetrics saved to: {metrics_file}")

        # Создаем сводную таблицу
        summary = []
        for model_name, metrics in results.items():
            summary.append({
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'] or 'N/A'
            })

        summary_df = pd.DataFrame(summary)
        summary_file = os.path.join(output_dir, f'summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)

        print(f"Summary saved to: {summary_file}")

        # Генерируем графики
        self.plot_results(results, output_dir, timestamp)

        return metrics_file, summary_file

    def plot_results(self, results, output_dir, timestamp):
        """Генерация графиков с результатами"""
        plt.style.use('seaborn-v0_8-darkgrid')

        # 1. Сравнение метрик моделей
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)

        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

        models = list(results.keys())

        for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx // 3, idx % 3]

            # Собираем значения только для моделей, у которых есть эта метрика
            model_names = []
            values = []
            colors = ['#3498db', '#2ecc71', '#e74c3c']

            for i, model in enumerate(models):
                if results[model][metric] is not None:
                    model_names.append(model)
                    values.append(results[model][metric])

            if values:  # Только если есть значения для построения
                bars = ax.bar(model_names, values, color=colors[:len(model_names)])
                ax.set_title(name)
                ax.set_ylim(0, 1)

                # Исправленная строка - используем FixedLocator
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names, rotation=45)

                # Добавляем значения на столбцы
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(name)

        # Убираем пустой subplot если нужно
        for idx in range(len(metrics_to_plot), 6):
            row = idx // 3
            col = idx % 3
            if row < 2 and col < 3:
                axes[row, col].axis('off')

        plt.tight_layout()
        plot_file = os.path.join(output_dir, f'comparison_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {plot_file}")


def interactive_testing(tester):
    """Интерактивное тестирование на своих текстах"""
    print("\n" + "=" * 60)
    print("INTERACTIVE TESTING")
    print("=" * 60)
    print("Enter text to classify (or 'quit' to exit):")

    while True:
        print("\n" + "-" * 40)
        text = input("\nEnter text: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if not text:
            print("Please enter some text!")
            continue

        print(f"\nText length: {len(text)} characters")

        # Предсказание
        results = tester.predict_single_text(text)

        print("\nPredictions:")
        for model_name, pred in results.items():
            if 'error' in pred:
                print(f"  {model_name}: ERROR - {pred['error']}")
            else:
                confidence = pred['confidence'] * 100 if pred['confidence'] else 'N/A'
                if pred['confidence_human'] is not None and pred['confidence_ai'] is not None:
                    print(f"  {model_name}: {pred['prediction']} "
                          f"(Human: {pred['confidence_human']:.1%}, "
                          f"AI: {pred['confidence_ai']:.1%}, "
                          f"Confidence: {pred['confidence']:.1%})")
                else:
                    print(f"  {model_name}: {pred['prediction']} (no confidence scores available)")


def main():
    """Основная функция тестирования"""
    print("=" * 60)
    print("ML MODELS TESTING SUITE")
    print("=" * 60)

    # Инициализируем тестер
    tester = ModelTester(models_dir='saved_models')

    # 1. Тестирование на тестовом наборе
    test_data, results = tester.evaluate_on_test_set()

    # 2. Сохраняем результаты
    metrics_file, summary_file = tester.save_results(results)

    # 3. Интерактивное тестирование
    interactive_testing(tester)

    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60)
    print(f"Results saved in: ../../results/ml_results/")
    print(f"Best model: {max(results.items(), key=lambda x: x[1]['accuracy'])[0]}")

    # Возвращаем результаты для дальнейшего анализа
    return test_data, results


if __name__ == "__main__":
    test_data, results = main()