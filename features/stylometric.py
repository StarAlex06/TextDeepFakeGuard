# features/stylometric.py
import re
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

# Условный импорт textstat - он опционален
try:
    import textstat

    # Проверяем доступные функции в textstat
    if hasattr(textstat, 'flesch_reading_ease'):
        from textstat import flesch_reading_ease, gunning_fog

        TEXTSTAT_AVAILABLE = True
    else:
        # В новых версиях могут быть другие названия
        TEXTSTAT_AVAILABLE = False
        print("Warning: textstat has different function names. Using fallback.")
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("Warning: textstat not installed. Using fallback for readability metrics.")

# Скачиваем необходимые ресурсы nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt_tab')


class StylometricFeatureExtractor:
    def __init__(self):
        self.feature_names = []

    def extract_all(self, text):
        """Извлечение всех стилометрических признаков"""
        features = {}

        # Базовые статистики длины
        features.update(self._length_features(text))

        # Лексическое разнообразие
        features.update(self._lexical_diversity_features(text))

        # Синтаксические особенности
        features.update(self._syntactic_features(text))

        # Стилистические особенности
        features.update(self._stylistic_features(text))

        # Сложность текста
        features.update(self._readability_features(text))

        self.feature_names = list(features.keys())
        return features

    def _length_features(self, text):
        """Признаки длины текста"""
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        chars = len(text)

        return {
            'char_count': chars,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': chars / max(len(words), 1),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'avg_syllables_per_word': self._avg_syllables_per_word(text)
        }

    def _lexical_diversity_features(self, text):
        """Лексическое разнообразие"""
        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        if not words:
            return {
                'type_token_ratio': 0,
                'hapax_legomena': 0,
                'entropy': 0,
                'simpsons_index': 0
            }

        word_counts = Counter(words)
        vocab_size = len(word_counts)
        total_words = len(words)

        # Type-Token Ratio (TTR)
        ttr = vocab_size / total_words

        # Hapax legomena (слова, встречающиеся 1 раз)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax / total_words

        # Энтропия
        probs = [count / total_words for count in word_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        # Индекс Симпсона
        simpsons = sum((count / total_words) ** 2 for count in word_counts.values())

        return {
            'type_token_ratio': ttr,
            'hapax_legomena': hapax_ratio,
            'entropy': entropy,
            'simpsons_index': simpsons
        }

    def _syntactic_features(self, text):
        """Синтаксические признаки"""
        words = word_tokenize(text)
        pos_tags = pos_tag(words) if words else []

        # Подсчет частей речи
        pos_counts = Counter(tag for word, tag in pos_tags)
        total_tags = len(pos_tags)

        if total_tags == 0:
            return {
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adj_ratio': 0,
                'adv_ratio': 0,
                'pronoun_ratio': 0,
                'function_word_ratio': 0
            }

        # Функциональные слова
        function_tags = {'DT', 'IN', 'CC', 'TO', 'PRP', 'PRP$', 'MD'}
        function_words = sum(1 for _, tag in pos_tags if tag in function_tags)

        return {
            'noun_ratio': (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) +
                           pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total_tags,
            'verb_ratio': (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) +
                           pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) +
                           pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_tags,
            'adj_ratio': (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) +
                          pos_counts.get('JJS', 0)) / total_tags,
            'adv_ratio': (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) +
                          pos_counts.get('RBS', 0)) / total_tags,
            'pronoun_ratio': (pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0)) / total_tags,
            'function_word_ratio': function_words / total_tags
        }

    def _stylistic_features(self, text):
        """Стилистические признаки"""
        # Знаки препинания
        punctuation = re.findall(r'[.,!?;:()\[\]{}"\'-]', text)
        words = word_tokenize(text)

        # Заглавные буквы
        uppercase = sum(1 for char in text if char.isupper())

        # Цифры
        digits = sum(1 for char in text if char.isdigit())

        total_chars = len(text)
        total_words = len(words)

        return {
            'punctuation_ratio': len(punctuation) / max(total_chars, 1),
            'uppercase_ratio': uppercase / max(total_chars, 1),
            'digit_ratio': digits / max(total_chars, 1),
            'exclamation_ratio': text.count('!') / max(total_words, 1),
            'question_ratio': text.count('?') / max(total_words, 1),
            'ellipsis_ratio': text.count('...') / max(total_words, 1)
        }

    def _readability_features(self, text):
        """Метрики читабельности"""
        try:
            if TEXTSTAT_AVAILABLE:
                # Старый стиль импорта
                flesch = flesch_reading_ease(text)
                gunning = gunning_fog(text)
            else:
                # Пробуем новые имена или вычисляем сами
                flesch = self._estimate_flesch(text)
                gunning = self._estimate_gunning_fog(text)
        except:
            flesch = 0
            gunning = 0

        return {
            'flesch_reading_ease': flesch,
            'gunning_fog_index': gunning
        }

    def _estimate_flesch(self, text):
        """Оценка индекса Флеша без textstat"""
        # Упрощенная формула
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        if len(sentences) == 0 or len(words) == 0:
            return 0

        total_syllables = sum(self._count_syllables(word) for word in words if word.isalpha())

        # Формула Флеша
        flesch = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))
        return flesch

    def _estimate_gunning_fog(self, text):
        """Оценка индекса Ганнинга"""
        sentences = sent_tokenize(text)
        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]

        if len(sentences) == 0 or len(words) == 0:
            return 0

        # Считаем сложные слова (с 3+ слогами)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)

        gunning = 0.4 * ((len(words) / len(sentences)) + 100 * (complex_words / len(words)))
        return gunning

    def _avg_syllables_per_word(self, text):
        """Среднее количество слогов на слово"""
        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        if not words:
            return 0

        total_syllables = 0
        for word in words:
            total_syllables += self._count_syllables(word)

        return total_syllables / len(words)

    def _count_syllables(self, word):
        """Подсчет слогов в слове"""
        word = word.lower()

        # Исключения
        if len(word) <= 3:
            return 1

        # Убираем окончания
        word = re.sub(r'(?:es|ed|ing)$', '', word)

        # Подсчет гласных групп
        vowels = 'aeiouy'
        count = 0
        prev_char_vowel = False

        for char in word:
            if char in vowels:
                if not prev_char_vowel:
                    count += 1
                prev_char_vowel = True
            else:
                prev_char_vowel = False

        return max(count, 1)