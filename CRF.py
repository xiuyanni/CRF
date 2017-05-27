"""
Conditional Random Fields (CRF)
"""

from __future__ import division
import numpy as np
import cPickle as pickle
from collections import defaultdict
from numpy import empty, zeros, ones, log, exp, sqrt, add, int32
from numpy.random import uniform
import itertools
from itertools import product

from scipy.optimize import fmin_l_bfgs_b

from utils import flatten, load_syn_data, sym_classifiers, save_theta
import random
from sklearn.model_selection import train_test_split

class CRF(object):
    def __init__(self, X, Y, binary_feature_type = "transition", regularity = 2, sigma = 1, random_seed = 100, verbose = False):
        '''
        :param X: the input matrix, np.array
        :param Y: corresponding labels, list
        :param binary_feature_type: specify the binary features, use default first

        '''
        self.binary_feature_type = binary_feature_type
        self.verbose = verbose
        self.X = np.array(X) # X: 160 * 100 * 8
        self.d = len(self.X[0][0]) # the feature size
        self.Y = np.array(Y) # Y: 160 * 100
        self.L = sorted(set(flatten(Y))) # a list of possibles labels
        self.K = len(self.L) # the number of all possible labels
        self.label_dict = {l: i for i, l in enumerate(self.L)} # map from label to ID
        self.unary_feature = None
        self.binary_feature = None
        self.Ys = list(product(self.L, repeat=len(X[0])))

        self.unary_feature_size = None
        self.binary_feature_size = None
        # use only when the binary feature type is transition
        self.random_seed = random_seed

        if regularity == 0:
            self.regularity = lambda w:0
            self.regularity_deriv = lambda w:0
        elif regularity == 1:
            self.regularity = lambda w: np.sum(np.abs(w)) / sigma
            self.regularity_deriv = lambda w:np.sign(w) / sigma
        else:
            v = sigma ** 2
            v2 = v * 2
            self.regularity = lambda w: np.sum(w ** 2) / v2
            self.regularity_deriv = lambda w: np.sum(w) / v

    def random_param(self):
        np.random.seed(self.random_seed)
        return np.random.randn(self.size())

    def size(self):
        # the size of unary feature, self.d * self.K is the size of binary features
        if self.binary_feature_type == "transition":
            return self.d * self.K + self.K * self.K
        if self.binary_feature_type == "concatenate":
            return self.K * self.d + self.K * self.d



    def size_edge(self):
        if self.binary_feature_type == "transition":
            return self.K * self.K
        if self.binary_feature_type == "concatenate":
            return self.K * self.d


    def get_features_for_sklearn(self):
        '''
        extract features for normal sklearn classifiers such as SVM,
        :return:
        '''
        return [x for sub_X in self.X for x in sub_X], self.Y.flatten()

    def label2dictid(self, f):
        # the map from a label to it's ID
        return self.label_dict[f]

    def get_unary(self, x, y):
        '''
        get the unary feature given x and y
        :param x: a single sample x:  8 * 1
        :param y: the corresponding label y
        :return:
        '''
        # for each single data
        assert isinstance(x, np.ndarray)
        uni_f = np.zeros((self.K, len(x)))
        label_id = self.label2dictid(y)
        uni_f[label_id] = x
        return np.array(uni_f).flatten()

    def get_unary_for_subX(self, subX, subY):
        '''
        get the unary feature vector
        :param subX: 100 * 8
        :param subY: 100
        :return:
        '''
        if self.verbose:
            print "unary subX: ", len(subX), len(subX[0])
        new_subX = map(lambda x: self.get_unary(x[0], x[1]), zip(subX, subY))

        if self.verbose:
            print "np.array(new_subX).shape", np.array(new_subX).shape
        new_subX_sum = np.sum(new_subX, axis=0)

        if self.verbose:
            print "get_unary_for_subX new_subX_sum: ", new_subX_sum.shape
        return new_subX_sum

    def get_binary(self, x1, y1, x2, y2):
        '''
        get the unary feature given x and y
        :param x: a single sample x:  8 * 1
        :param y: the corresponding label y
        :return:
        '''
        # for each single data
        assert isinstance(x1, np.ndarray)
        assert isinstance(x2, np.ndarray)
        assert len(x1) == len(x2)

        bi_f = np.zeros((self.K, len(x1)))
        label_id1 = self.label2dictid(y1)
        label_id2 = self.label2dictid(y2)
        bi_f[label_id1] = x1
        bi_f[label_id2] = x2
        return np.array(bi_f).flatten()



    def get_binary_for_subX(self, subX, subY):
        '''

        :param subX:
        :param subY:
        :return:
        '''
        assert len(subX) == len(subY)
        if self.binary_feature_type == "transition":

            transition = {(self.label2dictid(item[0]), self.label2dictid(item[1])): 0 for item in
                               sorted(itertools.product(self.L, self.L))}

            # print sorted(transition)
            if self.verbose:
                print "transition binary subX: ", len(subX), len(subX[0])

            for i in range(len(subY) - 1):
                label_id1 = self.label2dictid(subY[i])
                label_id2 = self.label2dictid(subY[i + 1])
                transition[(label_id1, label_id2)] += 1
            # print transition
            bi_f = np.array([transition[k] for k in sorted(transition)])
            if self.verbose:
                print "transition bi new_sub_X shape: ", bi_f.shape
            return bi_f

        #  new_subX = map(lambda x: self.get_unary(x[0], x[1]), zip(subX, subY))


        if self.binary_feature_type == "concatenate":
            bi_f = map(lambda x1, x2: self.get_binary(x1[0], x1[1], x2[0], x2[1]),
                           zip(subX, subY)[0:-1], zip(subX, subY)[1:])
            bi_f = np.array(bi_f)
            new_subX_sum = np.sum(bi_f, axis=0)
            if self.verbose:
                print "concatenate bi new_sub_X shape: ", new_subX_sum.shape
            return new_subX_sum

    def get_feature_vector(self, subX, subY):
        """

        :param subX:
        :param subY:
        :return:
        """
        unary_features = self.get_unary_for_subX(subX, subY)
        binary_features = self.get_binary_for_subX(subX, subY)
        feature_vector = np.array(list(unary_features) + list(binary_features))
        return feature_vector


    def get_feature_vectors(self):
        """
        concatenate the unary features and binary features
        :return:
        """
        feature_vectors = []
        for i in range(len(self.X)):
            subX = self.X[i]
            subY = self.Y[i]
            feature_vector = self.get_feature_vector(subX, subY)
            feature_vectors.append(feature_vector)
        return np.array(feature_vectors)

    def get_logZ(self, theta, subX):
        """

        :param theta:
        :param subX:
        :return:
        """
        # assert isinstance(theta, np.array)
        logZ = 0.0
        for subY in self.Ys:
            feature_vector = self.get_feature_vector(subX, subY)
            phi = np.dot(theta, feature_vector)
            logZ += exp(phi)
        logZ = log(logZ)
        return logZ

    def likelihood(self, theta):
        """

        :param theta:
        :return:
        """
        # fvs = self.get_feature_vectors()
        likelihood = 0.0
        for i in range(len(self.X)):
            subX = self.X[i]
            subY = self.Y[i]
            feature_vector = self.get_feature_vector(subX, subY)
            logZ = self.get_logZ(theta, subX)
            likelihood += np.dot(theta, feature_vector) - logZ

        return likelihood - self.regularity(theta)



    def gradient_likelihood(self, theta):
        """

        :param theta:
        :return:
        """
        grad = np.zeros(self.size())
        for i in range(len(self.X)):
            subX = self.X[i]
            subY = self.Y[i]
            feature_vector = self.get_feature_vector(subX, subY)

            p_total = np.zeros(self.size())
            for sub_y in self.Ys:
                fv = self.get_feature_vector(subX, sub_y)
                p_total += exp(np.dot(theta, fv) + 0.000001) * fv
            grad += feature_vector
            grad = grad / exp(log(self.get_logZ(theta, subX)))

        return grad - self.regularity_deriv(theta)

    def predict(self, theta, subX):

        max_marginal = float("-inf")
        target_labels = self.Ys[0]
        for subY in self.Ys:
            feature_vector = self.get_feature_vector(subX, subY)
            marginal = np.dot(theta, feature_vector)
            if marginal > max_marginal:
                target_labels = subY
        return target_labels

    def score(self, theta, X_test, y_test):
        correct = 0.0
        for i in range(len(X_test)):
            subX = X_test[i]
            subY = y_test[i]
            predicted = self.predict(theta, subX)
            correct += sum(1.0 for x, y in zip(subY, predicted) if x == y) / len(subY)
        return correct / (len(X_test) * len(X_test[0]))






def main():
    # test some functions
    # X, Y = load_syn_data('./data/result.mat')
    X, Y = load_syn_data("./data/crf_small_data.mat")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)


    crf = CRF(X_train, y_train)
    theta = crf.random_param()

    print "features:", crf.size()
    print "labels:", len(crf.L)

    #print "theta:", theta
    print "log likelihood:", crf.likelihood(theta)



    theta, fmin, _ = fmin_l_bfgs_b(lambda theta: crf.likelihood(theta), theta, fprime=lambda theta: crf.gradient_likelihood(theta),
                                    maxiter=10)

    save_theta(theta, "./model/brute_force_")

    print "score: ", crf.score(theta, X_test, y_test)



if __name__ == '__main__':
    main()


