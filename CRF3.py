# this version of code is extended from Nakatani Shuyo / Cybozu Labs Inc. work

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Conditional Random Field
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
# from scipy import maxentropy
from numpy import empty, zeros, ones, log, exp, sqrt, add, int32
from utils import load_syn_data, save_theta, read_ocr_data
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

import time

def logsumexp(a):
    """
    Compute the log of the sum of exponentials of an array ``a``, :math:`\log(\exp(a_0) + \exp(a_1) + ...)`
    """
    b = a.max()
    return b + log((exp(a-b)).sum())


def logdotexp_vec_mat(loga, logM):
    return numpy.array([logsumexp(loga + x) for x in logM.T], copy=False)

def logdotexp_mat_vec(logM, logb):
    return numpy.array([logsumexp(x + logb) for x in logM], copy=False)

def flatten(x):
    a = []
    for y in x: a.extend(flatten(y) if isinstance(y, list) else [y])
    return a

class FeatureVector(object):
    def __init__(self, features, xlist, ylist=None):
        '''intermediates of features (sufficient statistics like)'''
        flist = features.features_edge
        glist = features.features
        self.K = len(features.labels)
        # print "x_list", xlist
        # expectation of features under empirical distribution (if ylist is specified)
        if ylist:
            self.Fss = numpy.zeros(features.size(), dtype=float)
            for y1, y2 in zip(["start"] + ylist, ylist + ["end"]):
                self.Fss[:len(flist)] += [f(y1, y2) for f in flist]
            for y1, x1 in zip(ylist, xlist):
                self.Fss[len(flist):] += [g(x1, y1) for g in glist]
            # print "self.Fss: ", self.Fss


        # index list of ON values of edge features
        self.Fon = [] # (n, #f, indexes)

        # for calculation of M_i
        self.Fmat = [] # (n, K, #f, K)-matrix
        self.Gmat = [] # (n, #g, K)-matrix
        for x in xlist:
            mt = numpy.zeros((len(glist), self.K), dtype=int)
            for j, g in enumerate(glist):
                mt[j] = [g(x, y) for y in features.labels]  # sparse
            self.Gmat.append(mt)
            #self._calc_fmlist(flist, x) # when fmlist depends on x_i (if necessary)

        # when fmlist doesn't depend on x_i
        self._calc_fmlist(features)


    def _calc_fmlist(self, features):
        flist = features.features_edge
        fmlist = []
        f_on = [[] for f in flist]
        for k1, y1 in enumerate(features.labels):
            mt = numpy.zeros((len(flist), self.K), dtype=int)
            for j, f in enumerate(flist):
                mt[j] = [f(y1, y2) for y2 in features.labels]  # sparse
                f_on[j].extend([k1 * self.K + k2 for k2, v in enumerate(mt[j]) if v == 1])
            fmlist.append(mt)
        self.Fmat.append(fmlist)
        self.Fon.append(f_on)

    def cost(self, theta):
        return numpy.dot(theta, self.Fss)

    def logMlist(self, theta_f, theta_g):
        '''for independent fmlists on x_i'''
        fv = numpy.zeros((self.K, self.K))
        for j, fm in enumerate(self.Fmat[0]):
            fv[j] = numpy.dot(theta_f, fm)
        return [fv + numpy.dot(theta_g, gm) for gm in self.Gmat] + [fv]

class Features(object):
    def __init__(self, labels):
        self.features = []
        self.features_edge = []
        self.labels = ["start","stop"] + flatten(labels)

    def start_label_index(self):
        return 0
    def stop_label_index(self):
        return 1
    def size(self):
        return len(self.features) + len(self.features_edge)
    def size_edge(self):
        return len(self.features_edge)
    def id2label(self, list):
        return [self.labels[id] for id in list]

    def add_feature(self, f):
        self.features.append(f)
    def add_feature_edge(self, f):
        self.features_edge.append(f)

class CRF(object):
    def __init__(self, features, regularity, sigma=1):
        self.features = features
        if regularity == 0:
            self.regularity = lambda w:0
            self.regularity_deriv = lambda w:0
        elif regularity == 1:
            self.regularity = lambda w: numpy.sum(numpy.abs(w)) / sigma
            self.regularity_deriv = lambda w:numpy.sign(w) / sigma
        else:
            v = sigma ** 2
            v2 = v * 2
            self.regularity = lambda w: numpy.sum(w ** 2) / v2
            self.regularity_deriv = lambda w: numpy.sum(w) / v

    def random_param(self):
        return numpy.random.randn(self.features.size())

    def logalphas(self, Mlist):
        logalpha = Mlist[0][self.features.start_label_index()] # alpha(1)
        logalphas = [logalpha]
        for logM in Mlist[1:]:
            logalpha = logdotexp_vec_mat(logalpha, logM)
            logalphas.append(logalpha)
        return logalphas

    def logbetas(self, Mlist):
        logbeta = Mlist[-1][:, self.features.stop_label_index()]
        logbetas = [logbeta]
        for logM in Mlist[-2::-1]:
            logbeta = logdotexp_mat_vec(logM, logbeta)
            logbetas.append(logbeta)
        return logbetas[::-1]

    def likelihood(self, fvs, theta):
        '''conditional log likelihood log p(Y|X)'''
        n_fe = self.features.size_edge() # number of features on edge
        t1, t2 = theta[:n_fe], theta[n_fe:]
        stop_index = self.features.stop_label_index()

        likelihood = 0
        for fv in fvs:
            logMlist = fv.logMlist(t1, t2)
            logZ = self.logalphas(logMlist)[-1][stop_index]
            likelihood += fv.cost(theta) - logZ
        return likelihood - self.regularity(theta)

    def gradient_likelihood(self, fvs, theta):
        n_fe = self.features.size_edge() # number of features on edge
        t1, t2 = theta[:n_fe], theta[n_fe:]
        stop_index = self.features.stop_label_index()
        start_index = self.features.start_label_index()

        grad = numpy.zeros(self.features.size())
        for fv in fvs:
            logMlist = fv.logMlist(t1, t2)
            logalphas = self.logalphas(logMlist)
            logbetas = self.logbetas(logMlist)
            logZ = logalphas[-1][stop_index]

            grad += numpy.array(fv.Fss, dtype=float) # empirical expectation

            expect = numpy.zeros_like(logMlist[0])
            for i in range(len(logMlist)):
                if i == 0:
                    expect[start_index] += numpy.exp(logalphas[i] + logbetas[i+1] - logZ)
                elif i < len(logbetas) - 1:
                    a = logalphas[i-1][:, numpy.newaxis]
                    expect += numpy.exp(logMlist[i] + a + logbetas[i+1] - logZ)
                else:
                    expect[:, stop_index] += numpy.exp(logalphas[i-1] + logbetas[i] - logZ)
            for k, indexes in enumerate(fv.Fon[0]):
                grad[k] -= numpy.sum(expect.take(indexes))

            for i, gm in enumerate(fv.Gmat):
                p_yi = numpy.exp(logalphas[i] + logbetas[i+1] - logZ)
                grad[n_fe:] -= numpy.sum(gm * numpy.exp(logalphas[i] + logbetas[i+1] - logZ), axis=1)

        return grad - self.regularity_deriv(theta)

    def inference(self, fvs, theta):
        from scipy import optimize
        likelihood = lambda x:-self.likelihood(fvs, x)
        likelihood_deriv = lambda x:-self.gradient_likelihood(fvs, x)
        return optimize.fmin_bfgs(likelihood, theta, fprime=likelihood_deriv)

    def tagging(self, fv, theta):
        n_fe = self.features.size_edge() # number of features on edge
        logMlist = fv.logMlist(theta[:n_fe], theta[n_fe:])

        logalphas = self.logalphas(logMlist)
        logZ = logalphas[-1][self.features.stop_label_index()]

        delta = logMlist[0][self.features.start_label_index()]
        argmax_y = []
        for logM in logMlist[1:]:
            h = logM + delta[:, numpy.newaxis]
            argmax_y.append(h.argmax(0))
            delta = h.max(0)
        Y = [delta.argmax()]
        for a in reversed(argmax_y):
            Y.append(a[Y[-1]])

        return Y[0] - logZ, Y[::-1]

    def tagging_verify(self, fv, theta):
        '''verification of tagging'''
        n_fe = self.features.size_edge() # number of features on edge
        logMlist = fv.logMlist(theta[:n_fe], theta[n_fe:])
        N = len(logMlist) - 1
        K = logMlist[0][0].size

        ylist = [0] * N
        max_p = -1e9
        argmax_p = None
        while True:
            logp = logMlist[0][self.features.start_label_index(), ylist[0]]
            for i in range(len(ylist)-1):
                logp += logMlist[i+1][ylist[i], ylist[i+1]]
            logp += logMlist[N][ylist[N-1], self.features.stop_label_index()]
            print ylist, logp
            if max_p < logp:
                max_p = logp
                argmax_p = ylist[:]

            for k in range(N-1,-1,-1):
                if ylist[k] < K-1:
                    ylist[k] += 1
                    break
                ylist[k] = 0
            else:
                break
        return max_p, argmax_p



def main():

    import time
    print time.asctime()
    tt0 = time.time()

    # X, Y = load_syn_data("./data/crf_small_data.mat")
    X, Y = read_ocr_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
    # possible labels list



    features = Features(Y_train)

    for label in features.labels:
        for i in range(len(X[0][0])):
            features.add_feature(lambda x,  y, index = i, l=label: x[index] if y == l else 0)
    features.add_feature_edge( lambda y_, y: 0 )

    fvs = [FeatureVector(features, x, y) for x, y in zip(X, Y)]
    fv = fvs[0]
    print "fv:", fv

    text_fvs = [FeatureVector(features, X_test[i]) for i in range(len(X_test))] # text sequence without labels

    start = time.time()
    print "The time required for feature engineering ", start - tt0
    print time.asctime()

    crf = CRF(features, 2)
    theta = crf.random_param()

    print "features:", features.size()
    print "labels:", len(features.labels)

    #print "theta:", theta
    print "log likelihood:", crf.likelihood(fvs, theta)

    theta = crf.inference(fvs, theta)
    save_theta(theta, "./model/")

    y_pred = []
    for x_test in text_fvs:

        t0 = time.time()

        prob, ys = crf.tagging(x_test, theta)
        print "tagging:", prob, features.id2label(ys)
        yy = features.id2label(ys)[:-1]
        y_pred.append(yy)
        t1 = time.time()
        print "The time required for each sample: ", t1 - t0

    y_test_all = flatten(Y_test)
    y_pred_all = flatten(y_pred)

    print "score: ", accuracy_score(y_test_all, y_pred_all)
    end = time.time()
    print time.asctime()
    print "The time elapse: ", end - start

    #
    # #print "theta:", theta
    # prob, ys = crf.tagging(text_fv, theta)
    # print "tagging:", prob, zip(test_texts[0], test_labels[0], features.id2label(ys))

if __name__ == "__main__":
    main()

