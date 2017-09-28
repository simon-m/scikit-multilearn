import unittest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from skmultilearn.problem_transform import Homer
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class CCTest(ClassifierBaseTest):

    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        classifier = Homer(
            # classifier=SVC(probability=True), require_dense=[False, True])
            classifier=OneVsRestClassifier(SVC(probability=True)), k=2)

        self.assertClassifierWorksWithSparsity(classifier, 'sparse')
        self.assertClassifierPredictsProbabilities(classifier, 'sparse')

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        classifier = Homer(
            # classifier=SVC(probability=True), require_dense=[False, True]
            classifier=OneVsRestClassifier(SVC(probability=True)), k=2)

        self.assertClassifierWorksWithSparsity(classifier, 'dense')
        self.assertClassifierPredictsProbabilities(classifier, 'dense')

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        classifier = Homer(
            # classifier=GaussianNB(), require_dense=[True, True])
            classifier=OneVsRestClassifier(GaussianNB()), k=2)

        self.assertClassifierWorksWithSparsity(classifier, 'sparse')
        self.assertClassifierPredictsProbabilities(classifier, 'sparse')

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        classifier = Homer(
            # classifier=GaussianNB(), require_dense=[True, True])
            classifier=OneVsRestClassifier(GaussianNB()), k=2)

        self.assertClassifierWorksWithSparsity(classifier, 'dense')
        self.assertClassifierPredictsProbabilities(classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        classifier = Homer(
            # classifier=GaussianNB(), require_dense=[True, True])
            classifier=OneVsRestClassifier(GaussianNB()), k=2)

        self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
