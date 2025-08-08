import numpy as np


class NearestNeighborDistanceMetric:
    def __init__(self, metric, matching_threshold, budget=None):
        if metric != "cosine":
            raise ValueError("Only cosine metric supported.")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget and len(self.samples[target]) > self.budget:
                self.samples[target] = self.samples[target][-self.budget:]

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)), dtype=np.float32)
        for i, target in enumerate(targets):
            if target not in self.samples:
                cost_matrix[i, :] = 1.0
                continue
            target_features = np.array(self.samples[target])
            for j, feature in enumerate(features):
                similarity = np.dot(target_features, feature) / (
                    np.linalg.norm(target_features) * np.linalg.norm(feature) + 1e-6
                )
                cost_matrix[i, j] = 1 - similarity
        return cost_matrix
