import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

# Завантаження моделі Word2Vec
model_path = 'C:/Users/Liza/PycharmProjects/pythonProject1/fiction.cased.tokenized.word2vec.300d'  # Вкажіть шлях до вашого файлу
model = KeyedVectors.load_word2vec_format('C:/Users/Liza/PycharmProjects/pythonProject1/fiction.cased.tokenized.word2vec.300d', binary=False)



# Косинусна подібність
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Евклідова відстань
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


# Функція для аналогії
def word_analogy(word1, word2, word3, model):
    vector1 = model[word1]
    vector2 = model[word2]
    vector3 = model[word3]

    # Використовуємо відношення
    result_vector = vector2 - vector1 + vector3
    # Пошук найближчого слова
    result = model.similar_by_vector(result_vector, topn=1)
    return result[0]


# Візуалізація за допомогою PCA
def visualize_words(words, model):
    word_vectors = [model[word] for word in words]
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)

    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.text(reduced_vectors[i, 0] + 0.1, reduced_vectors[i, 1] + 0.1, word)
    plt.show()


# Приклад використання
words = ['Київ', 'Україна', 'Токіо']
visualize_words(words, model)
