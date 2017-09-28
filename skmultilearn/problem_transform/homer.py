import time

import numpy as np

from sklearn import base
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

import skmultilearn


class Tree:
    def __init__(self, classifier=None, labels=None,
                 parent=None, children=None):
        self.classifier = classifier
        if labels is None:
            self.labels = []
        else:
            self.labels = labels
        self.parent = parent
        if children is None:
            self.children = []
        else:
            self.children = children

    def __str__(self):
        return "Tree (" + ",".join(map(str, self.labels)) + ")"

    def __repr__(self):
        return self.__str__()

    def dump(self):
        print(self.__str__())
        for child in self.children:
            self.dump_(child, 0)

    def dump_(self, tree=None, depth=0):
        print("   " * depth + "|-- " + str(tree))
        for child in tree.children:
            self.dump_(child, depth=(depth + 1))

"""
tree = Tree(labels=[0])
tree.children.append(Tree(labels=[1], parent=tree))
tree.children.append(Tree(labels=[2], parent=tree))
tree.children[0].children.append(Tree(labels=[3], parent=tree.children[0]))
tree.children[0].children.append(Tree(labels=[4], parent=tree.children[0]))
tree.children[1].children.append(Tree(labels=[5], parent=tree.children[1]))
tree.dump()
"""


def random_partition(elts, k):
    n_elts = len(elts)
    subsets = [[] for i in range(k)]
    # or we could use permutations
    random_indexes = np.random.choice(range(n_elts), n_elts, replace=False)
    i = 0
    for ri in random_indexes:
        subsets[i % k].append(elts[ri])
        i += 1
    return subsets

"""
print(random_partition(np.arange(11), 3))
print(random_partition(np.arange(11), 4))
print(random_partition(np.arange(11), 5))
print(random_partition(np.arange(10), 5))
"""


def get_disjoint_label_sets(labels, k, parent=None):
    partition = random_partition(labels, k)
    return [(elt, parent) for elt in partition if elt]

"""
print(get_disjoint_label_sets(np.arange(10), 3, parent="parent"))
print(get_disjoint_label_sets(np.arange(11), 3, parent=None))
"""


def filter_data(X, y, allowed_label_indices):
    indices_to_keep = []
    for i, labels in enumerate(y):
        if np.any(labels[allowed_label_indices]):
            indices_to_keep.append(i)
    # if len(allowed_label_indices) > 1:
    return X[indices_to_keep, :], np.array(y[indices_to_keep, :][:, allowed_label_indices])
    # else:
    #     return X[indices_to_keep, :], np.array(y[indices_to_keep, :][:, allowed_label_indices].reshape(-1, 1))

"""
y = np.array([[0, 0, 1], [1, 1, 0], [1, 0, 1]])
X = np.reshape(np.arange(3), (-1, 1))
print(filter_data(X, y, [0]))
print(filter_data(X, y, [1]))
print(filter_data(X, y, [1, 2]))
"""


class HOMER(skmultilearn.base.MLClassifierBase):
    def __init__(self, base_classifier, k):
        self.base_classifier = base_classifier
        self.k = k
        self.root = None
        self.n_labels = None
        self.copyable_attrs = ['base_classifier', 'k']

    # simple sequential non-recursive proof of concept
    def fit(self, X, y):
        labels = list(range(y.shape[1]))  # 1..N
        self.root = Tree(self.base_classifier.fit(X, y), labels)
        self.n_labels = len(labels)
        self.root.labels = labels
        # partition the labels, recalling the parent set/node
        label_set_stack = get_disjoint_label_sets(labels, self.k, parent=self.root)  # 1..k, k..N

        while label_set_stack:
            # print("--\nlss %s" % label_set_stack)
            ls, parent = label_set_stack.pop()
            # print("ls %s" % ls)
            # only samples with corresponding active labels
            node_X, node_y = filter_data(X, y, ls)
            # print("X %s" % node_X)
            # print("y %s" % node_y)
            node_classif = base.clone(self.base_classifier)
            node_classif_model = node_classif.fit(node_X, node_y)
            # node_classif = None
            node = Tree(node_classif_model, ls)
            # not a leaf
            if len(ls) > 1:
                sub_label_set = get_disjoint_label_sets(ls, self.k, parent=node)
                label_set_stack.extend(sub_label_set)
            parent.children.append(node)
            # print(parent)
            # print(parent.children)
            # self.root.dump()
            # print("")

        return self

    def predict(self, X):
        return np.array([self.predict_(x) for x in X])

    def predict_(self, x):
        if self.root is None:
            raise NotFittedError("Call method fit() first")

        node_stack = [self.root]
        global_pred = np.zeros(self.n_labels)

        while node_stack:
            # ("\nns %s" % node_stack)
            node = node_stack.pop()
            # print("node %s" % node)
            node_pred = node.classifier.predict(x.reshape(1, -1))
            if node_pred.size > 1:
                node_pred = node_pred[0]
            else:
                node_pred = np.array([node_pred])

            # print("node_pred %s" % node_pred)
            # node_pred = np.random.choice([0, 1], len(node.labels), replace=True)
            pred_by_label = {lab: pred for lab, pred in zip(node.labels, node_pred)}
            # print(pred_by_label)


            # print(node.children)
            for child in node.children:
                for label in child.labels:
                    if pred_by_label[label]:
                        # print(child)
                        node_stack.append(child)
                        break
            if not node.children:
                # print("leaf")
                # print(node)
                # print(node_pred)
                global_pred[node.labels[0]] = node_pred
                # print(global_pred)

        return global_pred

    def dump(self):
        self.root.dump()


RANDOM_STATE_SEED = round(time.time() % 1 * 1e9)
np.random.seed(RANDOM_STATE_SEED)
print(RANDOM_STATE_SEED)

X = np.array([[0, 0, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 1, 1],
              [0, 1, 0, 0],
              [0, 1, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 0, 0, 0],
              [1, 0, 0, 1],
              [1, 0, 1, 0],
              [1, 0, 1, 1],
              [1, 1, 0, 0],
              [1, 1, 0, 1],
              [1, 1, 1, 0],
              [1, 1, 1, 1]])
y = np.transpose(np.vstack((X[:, 0] == X[:, 1], X[:, 2] == X[:, 3],
                            X[:, 0] == X[:, 3], X[:, 1] == X[:, 2])))
h = HOMER(DecisionTreeClassifier(), 2)
h_model = h.fit(X, y)
h_model.dump()
preds = h_model.predict(X)
assert(np.all(preds == y))
