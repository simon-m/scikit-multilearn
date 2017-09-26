import numpy as np


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
        return ",".join(map(str, self.labels))


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


def get_disjoint_label_sets(labels, k, parent=None):
    partition = random_partition(labels, k)
    return [(elt, parent) for elt in partition]


def filter_data(X, y, allowed_label_indices):
    indices_to_keep = []
    for i, labels in enumerate(y):
        if np.any(labels[allowed_label_indices]):
            indices_to_keep.append(i)
    return X[i, :], y[i, :]


def print_tree(tree, depth=0):
    print("   " * depth + "-- " + str(tree))
    for child in tree.children:
        print_tree(child, depth=(depth + 1))


class HOMER:
    def __init__(self, base_classifier, k):
        self.base_classifier = base_classifier
        self.k = k
        self.root = Tree()
        self.n_labels = None

    # simple sequential non-recursive proof of concept
    def fit(self, X, y):
        labels = list(range(y.shape[0]))  # 1..N
        self.n_labels = len(labels)
        self.root.labels = labels
        # partition the labels, recalling the parent set/node
        label_set_stack = get_disjoint_label_sets(labels, self.k, parent=None)  # 1..k, k..N

        while label_set_stack:
            print(label_set_stack)
            ls, parent = label_set_stack.pop()
            # only samples with corresponding active labels
            node_X, node_y = filter_data(X, y, ls)
            # node_classif = self.base_classifier.fit(node_X, node_y)
            node = Tree(node_classif, ls, parent, [])
            sub_label_set = get_disjoint_label_sets(ls, self.k, parent=node)
            # not a leaf
            if sub_label_set:
                label_set_stack.extend(sub_label_set)
            parent.children.append(node)

        return self

    def predict(self, x):
        node_stack = [self.root]
        global_pred = np.zeros(self.n_labels)
        while node_stack:
            node = node_stack.pop()
            node_pred = node.classifier.predict(x)
            for child in node.children:
                for label in child.labels:
                    if node_pred[label]:
                        node_stack.append(child)
                        break
            if not node.children:
                global_pred[node.labels[0]] = node_pred
        return global_pred







