import numpy as np
import scipy.io
from scipy.misc import logsumexp
import glob
import pickle
import os
from sklearn import cross_validation

def logdotexp_vec_mat(loga, logM):
    return np.array([logsumexp(loga + x) for x in logM.T], copy=False)

def logdotexp_mat_vec(logM, logb):
    return np.array([logsumexp(x + logb) for x in logM], copy=False)

def flatten(x):
    a = []
    for y in x: a.extend(flatten(y) if isinstance(y, list) else [y])
    return a



def get_unary(x, num_labels, label_id):
    '''
    get the unary feature given x and y
    :param x: a single sample x:  8 * 1
    :param y: the corresponding label y
    :return:
    '''
    # for each single data
    assert isinstance(x, np.ndarray)
    uni_f = np.zeros((num_labels, len(x)))
    uni_f[label_id] = x
    return np.array(uni_f).flatten()

def load_bof_data():
    """
    :return:
    """
    path = "./data/BOFData/*.mat"
    files = glob.glob(path)

    for f in [files[0]]:
        mat = scipy.io.loadmat(f)
        print mat.keys()
        print len(mat["BOF_tr_K"][0])
        print mat["BOF_tr_K"][0][29][0]
        print len(mat["BOF_tr_K"][0][29][0])

        print len(mat["BOF_tr_M"][0])
        print len(mat["BOF_tr_M"][0][29][0])

        print len(mat["BOF_te_M"][0])
        print len(mat["BOF_te_K"][0])

        print len(mat["label_te"][0][0])

        print len(mat["label_tr"][0][0])





def load_syn_data(filename):
    mat = scipy.io.loadmat(filename)
    print mat.keys()

    print "shape of X: ", mat['X'].shape
    print "shape of Y: ", mat['Y'].shape
    print "shape of L: ", mat['L'].shape

    print "shape of one element in X: ", np.array(mat['X'][0][0]).shape
    print "shape of one element in Y: ", np.array(mat['Y'][0][0][0]).shape

    #  iterate through X, mat['X'][i][0] for i in range(len(mat['X']))

    #  iterate through Y, mat['Y'][0][i][0] for i in range(len(mat['Y'][0]))
    # for i in range(len(mat['Y'][0])):
    #     print len(mat['Y'][0][i][0])

    data_X = []
    data_Y = []
    assert len(mat['X']) == len(mat['Y'][0])
    for i in range(len(mat['X'])):
        # len(mat['X']): 160, mat['X'][i][0]: 8 * 100
        temp = np.array(mat['X'][i][0]).T
        data_X.append(temp)
        # len(mat['Y'][0]): 160
        # mat['Y'][0][i][0]: 100
        labels = list(mat['Y'][0][i][0])
        labels = [str(i) for i in labels]
        data_Y.append(labels)
    # X: is a list of 160 data, each data has 100 components, each component is 8 length
    # like in the crf2.py, 160 corresponds to 4 sentences
    # 100 corresponds to word and info pairs
    # 8 correspoinds to the each word and info pair
    print "shape of load X: ", len(data_X), len(data_X[0]), len(data_X[0][0])
    print "shape of load Y: ",len(data_Y), len(data_Y[0])
    return data_X, data_Y


def sym_classifiers(X, Y):

    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.4, random_state=42)
    model = LinearSVC()
    # model.fit(X_train, y_train)
    # score = model.score(X_test, y_test)

    scores = cross_validation.cross_val_score(model, X, Y, cv=5)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    return scores.mean()


def save_theta(theta, filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    pickle.dump(theta, open(filepath+"theta_train.p", "wb"))


def read_theta(filepath):
    theta = pickle.load(open(filepath+"theta_train.p", "rb"))
    return theta


def read_ocr_data():
    f = open('./data/letter.data')
    # f = open('letter.data')

    feature = []
    letter_label = []
    word_label = []
    words = []
    word = []
    letter_labels = []

    for line in f:
        ele_list = line.split()
        # letter_label.append(ele_list[0])
        word_label.append(ele_list[3])
        ff = [int(f) for f in ele_list[6:]]
        feature.append(ff)
        next_id = ele_list[2]
        word.append(ff)
        letter_label.append(ele_list[1])
        if next_id == "-1":
            words.append(word)
            letter_labels.append(letter_label)
            letter_label = []
            word = []
    assert len(words) == len(letter_labels)
    return words, letter_labels


def get_features_for_sklearn(X, Y):
    '''
    extract features for normal sklearn classifiers such as SVM,
    :return:
    '''
    return [x for sub_X in X for x in sub_X], flatten(Y)



if __name__ == '__main__':
    # load_bof_data()
    # exit()

    # X, Y = load_syn_data("./data/crf_small_data.mat")

    X, Y = read_ocr_data()
    # print X[0], Y[0]

    Xs, Ys = get_features_for_sklearn(X, Y)
    print len(Xs), len(Ys)
    assert len(Xs) == len(Ys)
    print "The accuracy for Linear SVM is: ", sym_classifiers(Xs, Ys)


