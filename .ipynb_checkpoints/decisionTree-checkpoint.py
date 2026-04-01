from sklearn.metrics import accuracy_score
import numpy as np

class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, prediction_probs=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.prediction_probs = prediction_probs
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.labels_in_train = None

    def trainTree(self, X, Y):
        self.labels_in_train = np.unique(Y).astype(int)
        data = np.concatenate((X, Y.reshape(-1,1)), axis=1)
        self.tree = self._build_tree(data, depth=0)

    def _build_tree(self, data, depth):
        X, y = data[:, :-1], data[:, -1]
        probs = np.array([np.mean(y == lbl) for lbl in self.labels_in_train])

        if depth >= self.max_depth or len(data) <= self.min_samples_leaf or len(np.unique(y)) == 1:
            return TreeNode(prediction_probs=probs)

        best_feat, best_thresh, best_left, best_right = self._find_best_split(data)
        if best_feat is None:
            return TreeNode(prediction_probs=probs)

        node = TreeNode(feature_idx=best_feat, threshold=best_thresh, prediction_probs=probs)
        node.left = self._build_tree(best_left, depth+1)
        node.right = self._build_tree(best_right, depth+1)
        return node

    def _find_best_split(self, data):
        X, y = data[:, :-1], data[:, -1]
        num_features = X.shape[1]
        best_feat, best_thresh = None, None
        best_left, best_right = None, None
        max_gain = -1

        for i in range(num_features):
            thresholds = np.unique(X[:, i])
            for t in thresholds:
                left_mask = X[:, i] <= t
                right_mask = X[:, i] > t
                left_y, right_y = y[left_mask], y[right_mask]

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                gain = self._entropy(y) - (len(left_y)/len(y)*self._entropy(left_y) + len(right_y)/len(y)*self._entropy(right_y))
                if gain > max_gain:
                    max_gain = gain
                    best_feat = i
                    best_thresh = t
                    best_left = data[left_mask]
                    best_right = data[right_mask]

        return best_feat, best_thresh, best_left, best_right

    def _entropy(self, y):
        probs = np.array([np.mean(y == lbl) for lbl in self.labels_in_train])
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        while node.left is not None and node.right is not None:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return self.labels_in_train[np.argmax(node.prediction_probs)]

    def evaluate(self, X_train, Y_train, X_test=None, Y_test=None):
        train_preds = self.predict(X_train)
        train_acc = accuracy_score(Y_train, train_preds)

        print("Train Accuracy:", train_acc)

        if X_test is not None and Y_test is not None:
            test_preds = self.predict(X_test)
            test_acc = accuracy_score(Y_test, test_preds)

            confusion = np.zeros((3, 3))
            for i in range(len(Y_test)):
                confusion[int(Y_test[i].item()) - 1][int(test_preds[i].item()) - 1] += 1

            print("Test Accuracy:", test_acc)
            return confusion

        return np.zeroes(3, 3)