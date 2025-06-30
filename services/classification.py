import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

class MWPNN:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)

    def _gaussian_kernel(self, dist):
        return np.exp(-dist ** 2 / (2 * self.sigma ** 2))

    def predict(self, X_test):
        probs = []
        for x in X_test:
            distances = cdist([x], self.X_train, 'euclidean')[0]
            weights = self._gaussian_kernel(distances)

            class_scores = []
            for cls in self.classes:
                indices = np.where(self.y_train == cls)[0]
                class_weight = np.sum(weights[indices])
                class_scores.append(class_weight)

            probs.append(self.classes[np.argmax(class_scores)])
        return np.array(probs)