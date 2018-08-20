import dionysus as d
import numpy as np
from joblib import Parallel, delayed
import sys
import os
import random;
np.set_printoptions(precision=4)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold
import timeit
from igraph import *
import random
import glob
random.seed(42)
NUM_FEATURE = 1000; NORMALIIZATION = 1


# Learning filtration related
def load_graph(graph, debug='off'):
    # exptect label to be numpy.ndarry of shape (n,). However protein_data is different so have to handle it differently
    assert type(graph) == str
    import pickle
    import os
    import time
    import numpy as np
    GRAPH_TYPE = graph

    directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/LearningFiltration'
    make_direct(directory)
    inputFile = directory + '/graph+label'
    if os.path.isfile(inputFile):
        start = time.time()
        print('Loading existing files...')
        fd = open(inputFile, 'rb')
        (graphs_, labels_) = pickle.load(fd)
        print('Loading takes %s'%(time.time()-start))
        return (graphs_, labels_)

    print('Start Loading from dataset...')
    file = "/Users/admin/Documents/osu/Research/DeepGraphKernels/datasets/dataset/" + GRAPH_TYPE + ".graph"
    if not os.path.isfile(file):
        file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + GRAPH_TYPE + '.graph'
    f = open(file, 'r')
    data = pickle.load(f)
    graphs = data['graph']
    labels = data['labels']
    if debug=='on':
        print(graph),
        print type(labels),
        print np.shape(labels)
    if graph == 'protein_data':
        labels = labels.reshape(1113,1)
        labels = np.array([-1] * 663 + [1] * 450)
    elif graph == ('nci1' or 'nci109'):
        labels = np.sign(labels - 0.5)

    print('Finish Loading graphs')
    outputFile = directory + '/graph+label'
    fw = open(outputFile, 'wb')
    dataset = (graphs, labels)
    pickle.dump(dataset, fw)
    fw.close()
    print('Finish Saving data for future use')
    return (graphs, labels)

def graph_processing(g, beta = np.array([1,0,0,0])):
    # (g, fv_list) = add_function_value(g, fv='closeness_centrality')

    g = function_basis(g) # belong to pipe0
    g = fv(g, beta)       # belong to pipe1
    (g, fv_list) = add_function_value(g, fv_input='fv_test', edge_value='max') # belong to pipe1
    # fv_list.sort(reverse=True)


    return g
def convert(dgms_tuple):
    # convert tuple to pd
    import dionysus as d
    return d.Diagram(dgms_tuple)
def fv(g, beta, label='y'):
    # beta: [deg, ricci, fiedler, cc, label]
    # compute the linear combination(beta is weights) of different filtration function. Store it in fv_test
    import numpy as np
    assert type(beta) == np.ndarray
    if label=='y':
        assert len(beta) == 5
        # assert min(beta) >= 0
        # assert abs(np.sum(beta) -1) < 0.01
        # assert abs(np.linalg.norm(beta)-1) < 0.01 # no normalization
        assert 'deg' and 'ricciCurvature' and 'fiedler' and 'cc' and 'label' in g.node[list(g.nodes)[0]].keys()
        for n in g.nodes():
            g.node[n]['fv_test'] = beta[0] * g.node[n]['deg'] + beta[1] * g.node[n]['ricciCurvature'] \
                                   + beta[2] * g.node[n]['fiedler'] + beta[3] * g.node[n]['cc'] + beta[4] * g.node[n]['label']
    elif label=='no':
        assert len(beta) == 4
        # assert min(beta) >= 0
        assert abs(np.sum(beta) - 1) < 0.01
        assert 'deg' and 'ricciCurvature' and 'fiedler' and 'cc' in g.node[list(g.nodes)[0]].keys()  # didn't add label
        for n in g.nodes():
            g.node[n]['fv_test'] = beta[0] * g.node[n]['deg'] + beta[1] * g.node[n]['ricciCurvature'] \
                                   + beta[2] * g.node[n]['fiedler'] + beta[3] * g.node[n]['cc']
    return g
def format_matrixprl(matrix_prl):
    # format work. Input matrix_prl. Outpt (dist_matrix, idx_dict)
    n = np.shape(matrix_prl)[0]
    dist_matrix = np.zeros((n, n))
    idx_dict = {}
    for i in range(n):
        dist_matrix[i] = matrix_prl[i][0]
        idx_dict.update(matrix_prl[i][1])
    for i in range(n):
        for j in range(i):
            idx_dict[(i, j)] = idx_dict[(j, i)][::-1] # reverse a tuple

        # idx_dict = merge_two_dicts(idx_dict, matrix_prl[i][1])
    return ((dist_matrix + dist_matrix.T), idx_dict)
def serial_computing(dgms):
    # serial testing get_matrix_i function
    for i in range(len(dgms)):
        get_matrix_i(i, dgms)
def add_dgms(dgm1, dgm2):
    def convert(dgm):
        # convert dgm to a list of tuples(length 2)
        return [(np.float(p.birth), np.float(p.death)) for p in dgm]
    data = convert(dgm1) + convert(dgm2)
    # print(data)
    return d.Diagram(data)



def flip_dgm(dgm):
    for p in dgm:
        if np.float(p.birth) < np.float(p.death):
            print(np.float(p.birth), np.float(p.death))
        assert np.float(p.birth) >= np.float(p.death)
    data = [(np.float(p.death), np.float(p.birth)) for p in dgm]
    return d.Diagram(data)
# print_dgm(flip_dgm(super_dgms[0]))
def equal_dgm(dgm1, dgm2):
    import dionysus as d
    if d.bottleneck_distance(dgm1, dgm2)!=0:
        return True
    else:
        return False
def generate_graphs(version=2):
    # Return the list of all graphs with up to seven nodes named in the Graph Atlas.
    import networkx as nx
    if version==1:
        graphs = []
        n = len(nx.graph_atlas_g())
        for i in range(n):
            graphs.append([nx.graph_atlas(i)])
        return graphs
    elif version==2:
        graphs = nx.graph_atlas_g()
        graphs = [[g] for g in graphs]
        return graphs[1:]
def read_all_graphs(n=5):
    # get all non-isomorphic graphs of node n
    path = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/nauty/'
    import networkx as nx
    graphs = nx.read_graph6(path + 'graph' + str(n) + 'c.g6')
    graphs = [[graph] for graph in graphs]
    return graphs
def filtration_repr(beta):
    # input: numpy.array like [1,1,0,0,0]
    # ouput a string deg(0.5) + ricci(0.5)
    beta = beta / float(sum(beta))
    beta_dict = {0: 'deg', 1: 'ricci', 2: 'fiedler', 3: 'cc', 4:'label'}
    output = ''
    for i in range(5):
        if beta[i] > 1e-5:
            output += beta_dict[i] + '(' + str(beta[i]) + ') + '
    return output

# dionysus
def test_dionysus():
    import dionysus as d
    dgm = d.Diagram([(1,2), (3,4)])
    for i in range(100):
        print dgm[i]
    for p in dgm:
        print(p)
def bad_example():
    import dionysus as d
    dgm1 = d.Diagram([(1,2.07464)])
    dgm1 = d.Diagram([(1, 2.04287)])
    dgm2 = d.Diagram([(1,1.68001), (1,1.68001), (1,1.68001)]) # this one doesn't work
    dgm2 = d.Diagram([(1, 1.71035)])
    # dgm2 = d.Diagram([(1,1.68), (1,1.68), (1,1.68)]) # But this one works
    print  d.bottleneck_distance(dgm1, dgm2)
    print  d.bottleneck_distance_with_edge(dgm1, dgm2)
def computePD(i):
    # print('Process %s' % i)
    import dionysus as d
    import numpy as np
    np.random.seed(42)
    f1 = d.fill_rips(np.random.random((i + 10, 2)), 2, 1)
    m1 = d.homology_persistence(f1)
    dgms1 = d.init_diagrams(m1, f1)
    # return [(p.birth, p.death) for p in dgms1[1]]  # advice from Dmitriy
    return dgms1[1]
def test_get_dgms(n_jobs=1):
    from joblib import delayed, Parallel
    return Parallel(n_jobs=-1)(delayed(computePD)(i) for i in range(70))


# get the persistence vector
def gaussian(x, mu, sig):
    return np.exp(-np.power(x-mu,2)/(2 * np.power(sig,2) + 0.000001))
def data_interface(dgm):
    return {'data': [tuple(i) for i in dgm2diag(dgm)], 'max': 2, 'min': -2}
def rotate_data(data,super_check):
    """
    :param data:
    :return: a list of tuples
    """
    def rotate(x, y):
        return np.sqrt(2) / 2 * np.array([x + y, -x + y])
    def flip(x,y):
        return np.array([y,x])

    length = len(data)
    rotated = []; point = [0,0];
    for i in range(0,length,1):
        if super_check == True:
            data[i] = flip(data[i][0],data[i][1])
        point = rotate(data[i][0], data[i][1]);
        point = (point[0], point[1])
        rotated.append(point)
    return rotated
def draw_data(data, imax, imin):
    """
    :param data: a list of tuples
    :return: a dictionary: vector of length 1000
    """
    from matplotlib import pyplot as mp
    discrete_num = 500
    assert (imax>imin)
    # print(discrete_num)
    # distr = gaussian(np.linspace(-100, 100, 1000), 0, 10000)
    distr = np.array([0]*discrete_num)
    par = data
    # xmin = 10; xmax=0;
    # for i in range(len(data)):
    #     assert (data[i][1]-data[i][0]>=0)
    #     xmin = min(data[i][1]-data[i][0],xmin)
    #     xmax = max(data[i][1]-data[i][0],xmax)
    for x, y in par:
        mu = x; sigma = y/3.0
        distr = distr + y*gaussian(np.linspace(imin-1, imax+1, discrete_num), mu, sigma)
        # mp.plot(gaussian(np.linspace(-10,10,120), mu, sig) + gaussian(np.linspace(-10,10,120),mu+7, sig))
        # mp.plot(data(-3,3,100)[:,:])for i in range(1,10000):
    # print(i)
    # call_i_example(i,'ricci_edge','sub')

    # distr = distr/max(distr)
    # mp.plot(distr)
    # mp.show()
    return distr
def persistence_vector(dgm):
    ## here filtration only takes sub or super
    def vectorize(dgm):
        result = data_interface(dgm)
        data = result['data']; imax = result['max']; imin = result['min']
        data = rotate_data(data,super_check=False)
        vector = draw_data(data,imax,imin)
        vector = np.array(vector).reshape(1, len(vector))
        return vector

    return vectorize(dgm)
def persistence_vectors(dgms):
    import time
    start = time.time()
    n1 = len(dgms)
    n2 = np.shape(persistence_vector(dgms[0]))[1]
    X = np.zeros((n1, n2))
    from joblib import Parallel, delayed
    X_list = Parallel(n_jobs=-1)(delayed(persistence_vector)(dgms[i]) for i in range(len(dgms)))
    for i in range(n1):
        X[i] = X_list[i]
    print('persistence_vectores takes %s'%(time.time()-start))
    # return X / float(np.max(X))
    from sklearn.preprocessing import normalize
    X = normalize(X, norm='l2', axis=1, copy=True)
    return X

def read_neuron(id=3, key='direct_distance'):
    """
    Read data from NeuronTools
    :return: a list of tuples, its max and min
    """
    file = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/TruePDs/' + key + '/' + str(id) + '_*'
    file_directory = glob.glob(file)[0]
    file = open(file_directory,"r")
    data = file.read(); data = data.split('\n')[:-1];
    Data = []

    # this may need to change for different python version
    for i in range(0,len(data)):
        data_i = data[i].split(' ')
        data_i_tuple = (float(data_i[0]), float(data_i[1]))
        Data = Data + [data_i_tuple]
    assert len(data) == len(Data)

    # extract max and min
    tmp = np.array(Data).reshape(-1, 2)
    tmp_range = (tmp[:, 1] + tmp[:, 0]) / np.sqrt(2)
    if len(tmp_range) == 0:
        tmp_range = [0]
    assert (len(tmp_range) > 0)
    imax = np.amax(tmp_range)+0.01
    imin = np.amin(tmp_range)
    # imax = 85
    # imin = -85
    return {'data': Data, 'min':imin, 'max':imax}
def vectorize(id=0, key = 'direct_distance'):
    result = read_neuron(id, key)
    data = result['data'];
    imax = result['max'];
    imin = result['min']
    # print('before rotation')
    data = rotate_data(data, super_check=True)
    # print(data)
    # print('after rotation')
    vector = draw_data(data, imax, imin)
    vector = np.array(vector).reshape(1, len(vector))
    return vector

    V = vectorize(i)
    return V
def compute_vector(key='direct_distance', parallel='on'):
    from sklearn.preprocessing import normalize
    X = np.zeros((1,NUM_FEATURE))
    cls = [1,2,3,4,5,6,7]
    num = [710, 63, 10, 14, 20, 31, 420]
    idx = [range(1,711), range(711,774), range(774,784), range(784,798),
           range(798,818),range(818,849), range(849,1269)]
    if parallel == 'off':
        for n in range(1, 1269):
            X = np.concatenate((X, vectorize(n, key)), axis=0)
        X = X[1:]
    from joblib import delayed, Parallel
    if parallel == 'on':
        X= Parallel(n_jobs=-1)(delayed(vectorize)(i, key) for i in range(1, 1269))
        X = np.vstack(X)
    X = np.concatenate((X[1-1:711,:], X[849:1269,:]), axis=0)
    X = normalize(X, norm='l2', axis=NORMALIIZATION, copy=True)
    Y = np.array([1] * 710 + [7] * 420).reshape(1130, )
    return X,Y
def searchclf(i):
    import time
    from sklearn import svm
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report, accuracy_score
    for score in ['accuracy']:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=i)
        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, scoring='%s' % score, n_jobs=-1, verbose=1)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set is \n %s with score %s" % (
        clf.best_params_, clf.best_score_))
        print(clf.best_params_)
        print("Grid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print("Detailed classification report:\n")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))

def evaluate_clf(X, Y):
    from sklearn.model_selection import StratifiedKFold
    from sklearn import svm
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + 'neuron' + '/result/'
    accuracy = []
    n = 5
    for i in range(n):
        # after grid search, the best parameter is {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}
        clf = svm.SVC(kernel='rbf', gamma=0.1, C=400)
        # clf = svm.SVC(C=40, kernel='linear')
        from sklearn.model_selection import cross_val_score
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        cvs = cross_val_score(clf, X, Y, n_jobs=-1, cv=k_fold)
        print(cvs)
        acc = cvs.mean()
        accuracy.append(acc)
    accuracy = np.array(accuracy)
    print('mean is %s, std is %s '%(accuracy.mean(), accuracy.std()))




def viz_persistence_vector(dgms, Y, graph, beta, X, rf, X_flag='No'):
    import time
    np.set_printoptions(precision=4)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as mp

    start = time.time()
    for i in range(0, len(dgms)):

        if Y[i]==1:
            color = 'skyblue'
        elif Y[i]==-1:
            color = 'lightcoral'
        elif Y[i] == 0:
            color = 'lightgreen'
        elif Y[i] == 2:
            color = 'palegoldenrod'
        else:
            print('Not color set for %s'%Y[i])
            return
        if X_flag=='No':
            distr = persistence_vector(dgms[i])
            mp.plot(distr[0], c=color, alpha=1, linewidth=.4)
        elif X_flag=='Yes': # visualize X directly instead of computing X from dgms
            mp.plot(X[i], c=color, alpha=.5, linewidth=.5)
        # mp.ylim(0,10)
    mp.title(graph + ' ' + str(beta) + '\nrf:' + str(round(100 * rf[1])) + 'BL:' + str(baseline(graph)))
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/Viz_algorithm/' + graph + '/vector/persistence_vector/'
    make_direct(direct)
    mp.savefig(direct + str(beta), format='svg')
    mp.close()
    print('viz_persistence_vector takes %s'%(time.time()-start))
    # mp.show()

def persistence_repr(dgms):
    from joblib import Parallel, delayed
    X = Parallel(n_jobs=-1)(delayed(persistence_vector)(dgms[i]) for i in range(len(dgms)))
    n = len(dgms)
    data = np.zeros((n, 500))
    for i in range(n):
        data[i] = X[i]
    from sklearn.preprocessing import normalize
    data = normalize(data)
    return data
def clf_search_offprint(X, Y, random_state=2, print_flag='off'):
    import time
    start = time.time()
    from sklearn import svm
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report, accuracy_score

    tuned_parameters = [{'kernel': ['linear'], 'C': [ 0.1, 1, 10, 50, 100, 1000]},
                        {'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10], 'C': [0.1, 1, 10, 100,1000]}]
    for score in ['accuracy']:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=random_state)
        import time;
        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, scoring='%s' % score, n_jobs=-1, verbose=0)
        clf.fit(X_train, y_train)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        if print_flag == 'on':
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print("Detailed classification report:\n")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(accuracy_score(y_true, y_pred))
            print(classification_report(y_true, y_pred))
        print('clf_search_offprint takes %s'%(time.time()-start))
        return {'param': clf.best_params_, 'score': clf.best_score_}
def evaluate_best_estimator(grid_search_re, X, Y):
    print('Start evaluating the best estimator...')
    import time
    from sklearn import svm

    # start multiple cv
    param = grid_search_re['param']
    assert isinstance(param, dict)
    start = time.time()
    if len(param) == 3:
        clf = svm.SVC(kernel='rbf', C=param['C'], gamma = param['gamma'])
    else:
        clf = svm.SVC(kernel='linear', C=param['C'])
    # the easy cv method
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    # print('this is easy cv')
    import datetime
    now = time.time()
    cv_score = []
    n = 5
    for i in range(1, n):
        # print('this is cv %s' % i)
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        scores = cross_val_score(clf, X, Y, cv=k_fold, scoring='accuracy', n_jobs=-1)
        # print(scores)
        cv_score.append(scores.mean())
        # print("Accuracy: %0.3f (+/- %0.3f) \n" % (scores.mean(), scores.std() * 2))
    cv_score = np.array(cv_score)
    end = time.time()
    print('Evaluation takes %0.3f. After averageing %0.1f cross validations, the mean accuracy is %0.3f, the std is %0.3f\n' %(end-now, n, cv_score.mean(), cv_score.std()))


    return 0


# Auxliary function
def plot_test():
    import matplotlib.pyplot as plt
    ax = plt.subplot()
    ax.plot([1,2,3,4])
    plt.close()
    print('Test plt successfully')


def get_meshdata(start=87, end=177):
    f = open('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/landscape.txt')
    lines = f.readlines()
    lines = lines[start: end]
    lis = []
    for line in lines:
        new_line = line.replace('array', 'np.array')
        lis += [eval(new_line)]
    x = lis
    meshdata = {}
    for i in range(len(x)):
        assert x[i][0][2] == 0
        assert x[i][0][4] == 0
        x_ = x[i][0][1]
        y_ = x[i][0][3]
        z_ = x[i][0][0]
        meshdata[(x_, y_, z_)] = x[i][1]
    # print(type(meshdata), len(meshdata))
    return meshdata
meshdata = get_meshdata(start=195, end=285)
def viz_meshdata(graph):
    import matplotlib.pyplot as plt
    xy_ = np.array(meshdata.keys())
    x = xy_[:,0]
    y = xy_[:,1]
    z = xy_[:,2]
    t = meshdata.values()
    for i in range(len(x)):
        assert meshdata[x[i], y[i], z[i]] == t[i]
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(x, y, z, s=10, c=t)
    fig.colorbar(im)
    ax.set_xlabel('ricci'); ax.set_ylabel('cc'); ax.set_zlabel('deg')
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/Viz_algorithm/' + graph + '/landscape/'
    make_direct(direct)
    filename = 'viz_land.png'
    plt.savefig(direct+filename)
    plt.close()
# viz_meshdata('nci1')
def dump_data(GRAPH_TYPE, dataset, dataname, beta=-1, still_dump='yes', skip='yes'):
    if skip=='yes':
        return
    # save dataset in a directory for future use
    import pickle, os
    directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/LearningFiltration/'
    if (str(type(beta))=="<type 'numpy.ndarray'>") and (len(beta)==5):
        print('Saving in beta subdirectory...')
        beta_str = str(beta[0]) + '_' +  str(beta[1]) + '_' + str(beta[2]) + '_' + str(beta[3]) + '_' + str(beta[4])
        directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/LearningFiltration/' + beta_str + '/'
        make_direct(directory)

    outputFile = directory + dataname
    if os.path.exists(outputFile):
        if still_dump == 'no':
            print('File already exists. No need to dump again.')
            return
    print('Dumping...')
    fw = open(outputFile, 'wb')
    pickle.dump(dataset, fw)
    fw.close()
    print('Finish Saving data %s for future use'%dataname)

def load_data(GRAPH_TYPE, dataname, beta=-1, no_load='yes'):
    # load data from a directory
    import time
    start = time.time()
    if no_load == 'yes':
        return (None, 'Failure')
    import pickle, os
    directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/LearningFiltration/'
    if (str(type(beta))=="<type 'numpy.ndarray'>") and (len(beta)==5):
        beta_str = str(beta[0]) + '_' + str(beta[1]) + '_' + str(beta[2]) + '_' + str(beta[3]) + '_' + str(beta[4])
        directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/LearningFiltration/' + beta_str + '/'
        make_direct(directory)

    inputFile = directory + dataname
    if os.path.isfile(inputFile):
        print('Loading existing files %s...'%dataname)
        fd = open(inputFile, 'rb')
        dataset = pickle.load(fd)
        print('Loading takes %s'%(time.time()- start))
        return (dataset, 'success')
    else:
        return (None, 'Failure')
def print_best(best):
            for score in best:
                (c, sigma, train_acc, test_acc, test_std) = score[0], score[1], score[2], score[3], score[4]
                print('C: %s, sigma: %s, train_acc:%s test_acc:%s test_std:%s '%(c, sigma, train_acc, test_acc, test_std))
def baseline(graph):
    if graph=='mutag':
        return '83/87'
    if graph == 'ptc':
        return '54/63'
    if graph == 'protein_data':
        return '67/76'
    if graph == 'nci1':
        return '65/84'
    if graph == 'nci109':
        return '64/85'
    else:
        return 'NoBaseline'
def make_direct(direct):
    # has side effect
    import os
    if not os.path.exists(direct):
            os.makedirs(direct)
def kernel_summary(K):
    print('Min is %s Max is %s Average is %s'%( np.min(K), np.max(K), np.average(K)))
def convert2igraph(graph):
    from igraph import *
    keys = graph.keys()
    try:
        assert keys == range(len(graph.keys()))
    except AssertionError:
        print(keys, len(graph.keys()))
    gi = Graph()
    # add nodes
    for i in keys:
        gi.add_vertices(1)
    # add edges
    for i in keys:
        for j in graph[i]['neighbors']:
            if j>i:
                gi.add_edges([(i,j)])
    # add labels
    for i in keys:
        gi.vs[i]['id'] = 'Nonesense'
        gi.vs[i]['label'] = graph[i]['label'][0]
    # print(gi)
    # print (gi.vs[1])
    assert len(gi.vs) == len(graph.keys())
    return gi
def load_graph_igraph(graph):
    assert type(graph)==str
    import os
    GRAPH_TYPE = graph
    pass
    import pickle
    file = "/Users/admin/Documents/osu/Research/DeepGraphKernels/datasets/dataset/" + GRAPH_TYPE + ".graph"
    if not os.path.isfile(file):
        file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + GRAPH_TYPE + '.graph'
    f = open(file, 'r')
    data = pickle.load(f)
    data = data['graph']
    return data
def Kernel2Dist(K):
    import numpy as np
    # K = K / float(np.max(K))
    n = np.shape(K)[0]
    diag = np.diag(K).reshape(n,1)
    k1 = np.concatenate((diag, np.ones((n,1))), axis=1)
    k2 = np.concatenate((np.ones((1,n)), diag.reshape(1,n) ), axis=0)
    print(np.shape(k2))
    dist_matrix = np.dot(k1, k2)
    dist_matrix = dist_matrix - 2 * K
    print('Number of zeros before truncation %s'%np.sum(dist_matrix == 0))
    dist_matrix = dist_matrix * (dist_matrix>0)
    print('Number of zeros before truncation %s'%np.sum(dist_matrix == 0))
    return dist_matrix
def rfclf(X,Y):
    import numpy as np
    if np.shape(X)==(1,2):
        return
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    score1 = np.array([0, 0])
    # score1 = cross_val_score(clf1, X, Y, cv=10)
    # score1 = np.array([0,0])
    clf2 = RandomForestClassifier(n_estimators=40, max_features=40, max_depth=None, min_samples_split=2, random_state=0,n_jobs=-1)
    score2 = cross_val_score(clf2, X, Y, cv=10)
    print('Try Random Forest, n_estimators=40, max_features=40 ')
    print('Accuracy is %s'%score2.mean())
    return (score1.mean(), score2.mean())
def unzip_databundle(databundle):
    assert np.shape(databundle)[1] == 4
    graphs = [databundle[i][0] for i in range(len(databundle))]
    dgms = [databundle[i][1] for i in range(len(databundle))]
    sub_dgms = [databundle[i][2] for i in range(len(databundle))]
    super_dgms = [databundle[i][3] for i in range(len(databundle))]
    return (graphs, dgms, sub_dgms, super_dgms)
def graphiso(dgms_list, beta, graph_isomorphism='off'):
    if graph_isomorphism == 'on':
        distinct = []
        for lis in dgms_list:
            if lis not in distinct:
                distinct.append(lis)
        print(filtration_repr(beta), len(distinct))
        print('\n')
        # continue
def clf_pdvector(graph, (sub_dgms, super_dgms, dgms), beta, Y, pd_flag='False'):
    # pd vector classification
        if pd_flag == 'True':
            dgms_summary(dgms)
            X = np.concatenate((persistence_vectors(sub_dgms), persistence_vectors(super_dgms)), axis=1)
            print(np.shape(X))
            rf_ = rfclf(X, Y); print(rf_)
            viz_persistence_vector(dgms, Y, graph, beta, X, rf_, X_flag='Yes')
            n = len(dgms); dist_matrix = np.zeros((n, n))
            MDS_simple(graph, beta, dist_matrix, X, Y, rf_, rs=42, annotate='no', X_flag='yes')
            grid_search_re = clf_search_offprint(X, Y, random_state=2)
            evaluate_best_estimator(grid_search_re, X, Y)
            print('Finish calculating persistence diagrams')


# def normalize_kernel(K):
#     import numpy as np
#     print(np.min(K))
#     # assert np.min(K)>=0
#     # regularizer = float(0.001 * np.max(K))
#     K = K + np.eye(np.shape(K)[0]) * 1e-9
#     import numpy as np
#     assert np.shape(K)[0]==np.shape(K)[1]
#     v = np.diag(K)
#     v = np.sqrt(v) + 1e-10
#     print ('The decomposition difference is %s'%np.max((v ** 2 - np.diag(K))))
#     assert ((v**2 - np.diag(K)) < 0.001).all()
#     v = 1.0/v
#     v = np.reshape(v, ((len(v),1)))
#     assert np.shape(np.dot(v, v.T)) == np.shape(K)
#     normalized_kernel = np.multiply(K, np.dot(v, v.T))
#     print ('The symmetric difference is %s'%np.max(normalized_kernel - normalized_kernel.T))
#     # print (np.max(normalized_kernel, axis=1) == np.diag(normalized_kernel))
#     try:
#         assert (np.max(normalized_kernel, axis=1) == np.diag(normalized_kernel)).all()
#     except AssertionError:
#         print('The normalized kernel is bad')
#         print(np.sum(np.max(normalized_kernel, axis=1) == np.diag(normalized_kernel))),
#         print(np.shape(normalized_kernel))
#     # for i in range(np.shape(normalize_kernel)[0]):
#     return normalized_kernel
def normalize_kernel(K):
    assert np.shape(K)[0]==np.shape(K)[1]
    v = np.diag(K)
    v = np.sqrt(v)
    assert (v**2 - np.diag(K)).all() < 0.001
    v = 1.0/v
    v = np.reshape(v, ((len(v),1)))
    assert np.shape(np.dot(v, v.T)) == np.shape(K)
    return np.multiply(K, np.dot(v, v.T))
def gk_dist(graph):
    import numpy as np
    import graphkernels.kernels as gk
    graph = load_graph_igraph(graph)
    print(type(graph), len(graph))
    assert graph.keys() == range(len(graph))
    igraph_list=[]
    # serial version
    # for i in range(len(graph)):
    #     igraph_list = igraph_list + [convert2igraph(graph[i])]

    from joblib import Parallel, delayed
    igraph_list = Parallel(n_jobs=-1)(delayed(convert2igraph)(graph[i]) for i in range(len(graph)))

    assert len(igraph_list) == len(graph)
    K_wl_2 = gk.CalculateWLKernel(igraph_list, par = 3)
    print(''); kernel_summary(K_wl_2)
    assert np.min(K_wl_2) >= 0
    K_wl_2_after = normalize_kernel(K_wl_2)
    kernel_summary(K_wl_2_after)
    return (K_wl_2, K_wl_2_after, Kernel2Dist(K_wl_2))
    # return K_wl_2_after
def MDS_simple(graph, beta, dist_matrix, X, y, rf, rs=42, annotate='no', X_flag='no'):
    # input is the distance matrix
    # ouput: draw the mds 2D embedding

    import os
    import random; random.seed(rs)
    np.set_printoptions(precision=4)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    from sklearn import manifold

    if X_flag == 'no':
        mds = manifold.MDS(dissimilarity='precomputed', n_jobs=-1, random_state=rs, verbose=0)
        tsne = manifold.TSNE(metric='precomputed', verbose=0, random_state=rs)
        assert np.shape(dist_matrix)[0] == np.shape(dist_matrix)[1] == len(y)
        pos = mds.fit_transform(dist_matrix)
        tsne_pos = tsne.fit_transform(dist_matrix)
    elif X_flag == 'yes':
        mds = manifold.MDS( n_jobs=-1, random_state=rs, verbose=0)
        tsne = manifold.TSNE(verbose=0, random_state=rs)
        try:
            pos = mds.fit_transform(X)
        except ValueError: # for nci1 closeness_centrality there is some error (ValueError: Array must be symmetric)
            print('There is ValueError for mds.fit')
            pos = np.zeros((np.shape(X)[0], 2))
        tsne_pos = tsne.fit_transform(X)
        pos = np.array(pos)

    for i in range(np.shape(tsne_pos)[0]):
        if tsne_pos[i,0] > -0.05:
            pass
            # break
            # print('%s,'%i),

    y_color = [color_map(i) for i in y]
    print(y_color.count('r'))
    y_color = np.array(y_color)
    assert len(y_color) == len(y)
    assert np.shape(pos)[0] == len(y); assert np.shape(tsne_pos)[0] == len(y)

    # start to plot
    fig = plt.figure()
    plt.axis('off')
    plt.title(graph + ' ' + str(beta) + '\nrf:' + str(round(100 * rf[1])) + 'BL:' + str(baseline(graph)))
    ax = fig.add_subplot(211)
    ax.scatter(np.array(pos[:, 0]), np.array(pos[:, 1]), c=y_color, s=2)
    # ax.set_ylim(-1, 1)
    # ax.set_xlim(-1, 1)
    from textwrap import wrap
    fig.subplots_adjust(top=0.8)
    for xi, yi, pidi in zip(pos[:, 0], pos[:, 1], range(len(y))):
        if (annotate == 'yes') and (pidi % 30 == 0):
            plt.annotate(str(pidi), xy=(xi, yi), xytext=(xi, yi))  # xytext=(xi+0.05 * np.random.rand(), yi+0.05 * np.random.rand())

    ax = fig.add_subplot(212)
    ax.scatter(tsne_pos[:, 0], tsne_pos[:, 1], c=y_color, s=1.2)
    for xi, yi, pidi in zip(tsne_pos[:, 0], tsne_pos[:, 1], range(len(y))):
        if (annotate == 'yes') and (pidi % 30 == 0):
            plt.annotate(str(pidi), xy=(xi, yi), xytext=(xi, yi))  # xytext=(xi+0.05 * np.random.rand(), yi+0.05 * np.random.rand())

    direct = './Viz_algorithm/' + graph + '/vector/'
    make_direct(direct)
    filename = '/vector_' + str(beta) + '.svg'
    fig.savefig(direct + filename, format='svg')
    plt.close()
    print('Saving figure successfully')


def sample_data(data, Y, n_sample=200):
    import random
    n = len(Y)
    assert n > n_sample
    idx = random.sample(range(1, n), n_sample)
    data = {k:data[k] for k in tuple(idx) if k in data}
    Y = Y[0:200]
    return (data, Y)
def beta_update(beta ,gradient, step=1e-1, normalization='yes'):
    print('Step size is %s'%step)
    print('Before update: beta is %s'%beta)
    print('The gradient is %s'%gradient)
    beta =  (beta - step * gradient)
    if normalization == 'no':
        print('After update: beta is %s\n' %(beta))
        return  beta
    elif normalization == 'yes':
        print('After update: beta is %s\n' %(beta/np.linalg.norm(beta)))
        return  beta/np.linalg.norm(beta)
    # return beta/np.sum(beta)
def print_dgm(dgm):
    for p in dgm:
        print(p),
def scatter_dgm(dgm):
    scatter = []
    for p in dgm:
        scatter += [(p.birth, p.death)]
    x = [i[0] for i in scatter]
    y = [i[1] for i in scatter]
    return scatter, x, y
def square(i):
    import dionysus as d
    import numpy as np
    f1 = d.fill_rips(np.random.random((i + 10, 2)), 2, 1)
    m1 = d.homology_persistence(f1)
    dgms1 = d.init_diagrams(m1, f1)
    return dgms1[1]
def random_dgm(n):
    # generate ramdom dgm of cardinality n
    import numpy as np
    import dionysus as d
    lis = []
    for i in range(n):
        a = np.random.rand()
        b = np.random.rand()
        lis.append((a, a+b))
    return d.Diagram(lis)
def diag2dgm(diag):
    import dionysus as d
    if type(diag) == list:
        diag = [tuple(i) for i in diag]
    elif type(diag) == np.ndarray:
        diag = [tuple(i) for i in diag] # just help to tell diag might be an array

    dgm = d.Diagram(diag)
    return dgm
def distinct_list(lis):
    # input: a list of lists
    # output: a list of distinct lists
    distinct = []
    for i in lis:
        # print(i)
        if (list(i) not in distinct):
            distinct.append(list(i))
    return distinct
def normalize_beta(beta):
    return beta / float(np.sum(beta))
def generate_beta(n, k=5):
    # n is the length here
    # deg, ricci, fiedler, cc, label
    import itertools
    import numpy as np
    lst = list(itertools.product(range(k), repeat=n))
    # print(lst)
    weight = [np.array(tup)/float(sum(tup)) for tup in lst if (tup[0]==0 and tup[2]==0 and tup[4]==0)] # tup[4]==0 and tup[3]==0
    weight = distinct_list(weight)
    weight = [np.array(i) for i in weight]
    weight = sorted(weight, key=lambda x: x[1]) # sort by ricci
    return weight[1:] # remove [nan, nan, nan]
# weight = generate_beta(5)

def draw_landscape(graph):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    # Z = np.sin(R)
    Z = np.random.rand(40,40)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/Viz_algorithm/' + graph + '/landscape/';
    filename = 'img.png'
    plt.savefig(direct+filename)

draw_landscape('nci1')
def beta_mesh():
    mesh={}
    X = np.arange(-1,1,0.2)
    Y = np.arange(-1, 1, 0.2)
    for i in range(len(X)):
        for j in range(len(Y)):
            mesh[(i,j)] = np.array([0, X[i], 0, Y[j], 0])
    return mesh
def dgm2diag(dgm):
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    # print(diag)
    return diag
def random_lists(n):
    import numpy as np
    import dionysus as d
    lis = []
    for i in range(n):
        a = np.random.rand()
        b = np.random.rand()
        lis.append([a, a + b])
    return lis
def write_landscapedata(graph, new_data):
    assert type(new_data) == dict
    assert 'beta' in new_data.keys()
    assert 'others' in new_data.keys()

    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/Viz_algorithm/' + graph + '/landscape/';
    make_direct(direct)
    filename = 'landscape_data.txt'
    f = open(direct + filename, 'a+')
    f.write(str(new_data))
    f.write('\n')
    # f.close()



# random_lists(5)
def n_runs(graph):
    if (graph == 'nci1') or (graph == 'nci109'):
        return 10
    elif (graph == 'reddit_5K') or (graph == 'reddit_12K'):
        return 10
    else:
        return 100
def progressbar():
    from progress.bar import Bar
    import time
    bar = Bar('Processing', max=10)
    for i in range(20):
        time.sleep(.1)
        bar.next()
    bar.finish()
def dgms_summary(dgms):
    n = len(dgms)
    total_pts = [-1]*n
    for i in range(len(dgms)):
        total_pts[i] = len(dgms[i])
    print('Mean: %s, Std: %s, Min: %s, Max: %s'%(np.mean(total_pts), np.std(total_pts), np.min(total_pts), np.max(total_pts)))
def threshold_data(data, flag='off'):
    return 1
    # only print classify that yields reasonable good test accuracy
    if flag=='on':
        return 0
    if data == 'mutag':
        threshold = 78
    elif data == 'ptc':
        threshold = 55
    elif data == 'protein_data':
        threshold = 50
    elif data == ('nci1' or 'nci109'):
        threshold = 60
    return threshold
def onehot(n):
    import numpy as np
    y = np.zeros((2*n,2))
    for i in range(n):
        y[i][0] = -1
    for i in range(n, 2*n):
        y[i][1] = 1
    return y
onehot(2)
def hinge_loss(y, y_hat):
    assert ((y==1) or (y==-1))
    if y == 1:
        return max(0, 1-y_hat)
    elif y== -1:
        return max(0, 1+y_hat)
def hinge_gradient(y, y_hat):
    # scalar version. The gradient of hinge loss(1-y_hat*y)+
    assert (y == 1) or (y == -1)
    if y == 1:
        if y_hat >= 1:
            return 0
        elif y_hat < 1:
            return -1
    elif y == -1:
        if y_hat >= -1:
            return 1
        elif y_hat < -1:
            return 0
# hinge_gradient(-1, -1.0003468881649533)
class parallel():
    def get_dgms_pl(self):
        from multiprocessing import Pool
        if True:
            pool = Pool(processes=2)
            list_start_vals = range(20)
            array_2D = pool.map(square, list_start_vals)
            pool.close()
            return array_2D
    def get_dgms_pl2(self, n_jobs=1):
        from joblib import delayed, Parallel
        return Parallel(n_jobs=n_jobs)(delayed(square)(i) for i in range(10))
class Nodeid_Unfound_Error(Exception):
    pass
class pipeline0_Error(Exception):
    pass

# for i in range(100):
#     print d.bottleneck_distance_with_edge(random_dgm(150), random_dgm(150))

##############################

class test():
    def test_kernelMachine(self):
        import numpy as np
        # X = np.array([[-1, -1], [-12, -2], [1, 1.1], [2, 1], [4, 8]])
        # y = np.array([-1, -1, 1, 1, 1])
        X = np.random.rand(100,2)
        Y = np.array([-1]*50 + [1]*50)
        n = len(Y)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, range(n), test_size=0.1, random_state=43)
        n_train = len(y_train)
        from scipy.spatial.distance import pdist, cdist, squareform
        # kernel = pdist(X, 'euclidean')
        # kernel = squareform(kernel)
        # kernel = np.exp(np.multiply(kernel, -kernel))
        kernel = np.dot(X, X.T)
        kernel_train = kernel[np.ix_(indices_train, indices_train)]
        assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train) == True
        kernel_test = kernel[np.ix_(indices_test, indices_train)]
        np.shape(kernel_test)

        from sklearn.svm import SVC, SVR
        clf = SVC(kernel='linear', gamma=1, verbose=True)
        clf.fit(X_train, y_train)
        alpha = clf.dual_coef_
        alpha = padding_zero(alpha, n_train, clf.support_)
        # alpha = np.multiply(alpha, y_train)
        print('Number of support vector is %s' % clf.n_support_)
        dual_regression = np.dot(kernel_test, alpha) + clf._intercept_
        primal_regression = (np.dot(X_test, clf.coef_.T) + clf.intercept_).T # clf.coef_ is only available to linear kernel
        # print('Alpha is %s' % alpha)
        # print('My Regression is %s' % regression)
        print('Model Prediction is %s\n' % clf.predict(X_test))

        print('My dual Classification is %s' % np.sign(dual_regression))
        print('My dual primal classification is %s' % np.sign(primal_regression))
        print('Primal prediction is %s\n' % (np.sign(primal_regression) == clf.predict(X_test)))

        print('My dual Prediction is %s' % regression)
        print('My primal predication is %s'%(primal_regression))
        print('Decision boundary is %s'%clf.decision_function(X_test))

        print('Train accuracy is %s \n'%clf.score(X_train,y_train))


        ###########
        import numpy as np
        from sklearn.svm import SVC, SVR
        from scipy.spatial.distance import pdist, squareform
        from sklearn.model_selection import train_test_split

        X = np.random.rand(100, 2)
        Y = np.array([-1] * 50 + [1] * 50)
        n = len(Y)

        clf = SVC(kernel='precomputed', verbose=True)
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, range(n), test_size=0.1, random_state=43)
        n_train = len(y_train)
        kernel = pdist(X)
        kernel = squareform(kernel)
        kernel = np.exp(np.multiply(kernel, -kernel))
        kernel_train = kernel[np.ix_(indices_train, indices_train)]
        assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train) == True
        kernel_test = kernel[np.ix_(indices_test, indices_train)]

        clf.fit(kernel_train, y_train)
        alpha = clf.dual_coef_
        alpha = padding_zero(alpha, n_train, clf.support_)
        dual_regression = np.dot(kernel_test, alpha) + clf._intercept_
        # print('Alpha is %s' % alpha)
        print('My Dual Regression is %s' % dual_regression)
        print('Model Classification is %s' % clf.predict(kernel_test))
        print('Model Prediction is %s' % clf._decision_function(kernel_test).T)
        print('Dual regression - Model prediction: %s'%(dual_regression - clf._decision_function(kernel_test).T))
        print('Intercept is %s'%clf.intercept_)
        ###############
        from sklearn.metrics import accuracy_score
        lbd = 1e-7
        alpha = compute_alpha_kernel(kernel_train, y_train, lbd)
        K = kernel_test
        assert np.shape(K)[1] == len(alpha)
        y_pred = np.dot(kernel_test, alpha)
        if debug == 'on':
            print('The regression in belkin model is %s' % y_pred)
        y_reg = y_pred  # get the r_reg for gradient descent
        y_pred = np.sign(y_pred)

        y_pred_train = np.sign(np.dot(kernel_train, alpha))
        print('Alpha is %s' % alpha)
        print('My Regression is %s' % y_pred)
        print('Train accuracy is %s' % accuracy_score(y_train, y_pred_train))
        print('Test accuracy is %s' % accuracy_score(y_test, y_pred))
    def random_fit(self):
        def test_kernelMachine(self):
            import numpy as np
            X = np.random.rand(1000, 2)
            Y = np.array([-1] * 500 + [1] * 500)
            n = len(Y)

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, range(n),
                                                                                             test_size=0.01,
                                                                                             random_state=43)
            n_train = len(y_train)
            from scipy.spatial.distance import pdist, cdist, squareform
            kernel = pdist(X, 'euclidean')
            kernel = squareform(kernel)
            kernel = np.exp(np.multiply(kernel, -kernel))
            # kernel = np.dot(X, X.T)
            kernel_train = kernel[np.ix_(indices_train, indices_train)]
            assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train) == True
            kernel_test = kernel[np.ix_(indices_test, indices_train)]
            np.shape(kernel_test)

            from sklearn.svm import SVC, SVR
            from sklearn.metrics import accuracy_score
            clf = SVC(kernel='linear', gamma=1, verbose=True)
            clf.fit(X_train, y_train)
            print('Traing accuracy is %s'%accuracy_score(clf.predict(X_train), y_train))

    def test_wd(self):
        import dionysus as d
        import numpy as np
        f1 = d.fill_rips(np.random.random((20, 2)), 2, 1)
        m1 = d.homology_persistence(f1)
        dgms1 = d.init_diagrams(m1, f1)
        f2 = d.fill_rips(np.random.random((20, 2)), 2, 1)
        m2 = d.homology_persistence(f2)
        dgms2 = d.init_diagrams(m2, f2)
        wdist = d.wasserstein_distance(dgms1[1], dgms2[1], q=2)
        print("2-Wasserstein distance between 1-dimensional persistence diagrams:", wdist)
        bdist = d.bottleneck_distance(dgms1[1], dgms2[1])
        print("Bottleneck distance between 1-dimensional persistence diagrams:", bdist)

    def test_wd_symmetric(self):
        import dionysus as d
        dgm1 = d.Diagram([(5.5, 4.68182), (8.04545, 4.59091)])
        dgm2 = d.Diagram([(5.96, 3.48), (5.24, 4.04), (5.24, 4.04), (4.44, 3.48)])
        print (d.bottleneck_distance(dgm1, dgm2))
        print (d.bottleneck_distance(dgm2, dgm1))
        print(d.bottleneck_distance())

    def test_bd_with_edge(self):
        import dionysus as d
        # dgm1 = d.Diagram([(5.5, 4.68182), (8.04545, 4.59091)])
        # dgm2 = d.Diagram([(5.96, 3.48), (5.24, 4.04), (5.24, 4.04), (4.44, 3.48)])
        # dgm1 = d.Diagram([(4.68182, 5.5), (4.59091, 8.04545)])
        # dgm2 = d.Diagram([(3.48, 5.96), (4.04, 5.24), (4.04, 5.24), (3.48, 4.44)])

        dgm1 = d.Diagram([(1.1, 0)])
        dgm2 = d.Diagram([(0.1, 0)])
        print (d.bottleneck_distance_with_edge(dgm1, dgm2))


# Viz graph
def peekgraph(g, key='fv'):
    # input is nx.graph, output is the node and its node value
    for v, data in sorted(g.nodes(data=True), key=lambda x: x[1][key], reverse=True):
        print (v, data)
# peekgraph(graphs[77][1])
def draw_graph(g):
    # input is nx.graph
    import matplotlib.pyplot as plt
    import networkx as nx
    nx.draw(g)
    plt.draw()
    plt.show()
# draw_graph(graphs[77][1])
def viz_kernel(k):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # Make an array with ones in the shape of an 'X'
    a = k

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    # Bilinear interpolation - this will look blurry
    ax1.imshow(a, cmap=cm.Greys_r)

    ax2 = fig.add_subplot(122)
    # 'nearest' interpolation - faithful but blocky
    ax2.imshow(a, interpolation='nearest', cmap=cm.Greys_r)

    plt.show()

# MDS related
def color_map(i):
    if i == 1:
        return 0.1
    if i == 0:
        return 'b'
    if i == 2:
        return 0.6
    if i == 3:
        return 0.9
    elif i == -1:
        return 'r'
def new_label(graph):
    if graph == 'mutag':
        label = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 20, 21, 22, 23, 26, 29, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 52, 55, 56, 58, 60, 61, 64, 65, 68, 71, 72, 76, 78, 79, 81, 82, 83, 85, 87, 88, 91, 94, 99, 100, 102, 103, 104, 105, 107, 108, 111, 112, 113, 114, 116, 118, 119, 120, 123, 124, 127, 142, 145, 150, 152, 153, 154, 155, 156, 161, 162, 163, 164, 165, 167, 169, 170, 183, 186, 187, 153
] # [0, 0.5, 0, 0.5, 0] x>0
    if graph == 'protein_data':
        # label = [155,139,102,95,122,6,110, 26, 88,126,93,117,
        #         157,99,24,72,8,24,99,72,23,46,55,43,44,56,155, 22, 82, 40, 5,55,36,26, 34,15,46,
        #         103,41,21,87,38]
        # label = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 14, 16, 17, 20, 21, 23, 27, 28, 29, 31, 33, 34, 36, 37, 39, 42, \
    # 47, 49, 52, 58, 60, 61, 63, 64, 65, 66, 68, 69, 71, 75, 76, 77, 78, 79, 81, 83, 85, 91, 94, 96, 98, 100, 102, 104, \
    # 105, 107, 108, 111, 112, 113, 114, 116, 118, 119, 120, 123, \
    # 124, 125, 135, 136, 137, 138, 141, 145, 147, 150, 151, 152, 153, 155, 156, 160, \
    # 161, 162, 163, 164, 165, 167, 169, 170, 171, 176, 179, 183, 186, 187, 35] # for [1, 0, 0, 0, 0]
    #     label = [1, 4, 7, 9, 10, 11, 12, 14, 16, 18, 19, 20, 25, 27, 28, 29, 30, 31, 32, 33, 35, 39, 42, 45, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 83, 84, 85, 86, 89, 90, 91, 92, 96, 97, 98, 101, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 121, 123, 124, 125, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 98
# ] # for [0.7611 0.6486 0.     0.     0.    ]

        # label = [1, 2, 3, 4, 8, 9, 10, 11, 12, 14, 15, 16, 19, 21, 23, 24, 28, 29, 30, 31, 32, 33, 35, 36, 38, 40, 41, 43, 44, 45, 46, 47, 48, 50, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64, 65, 67, 69, 70, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 93, 94, 95, 97, 98, 99, 101, 102, 103, 105, 107, 108, 109, 110, 112, 113, 114, 116, 117, 118, 121, 124, 125, 126, 127, 128, 131, 132, 133, 135, 136, 140, 141, 144, 145, 146, 147, 148, 150, 151, 152, 155, 158, 163, 164, 165, 166, 168, 169, 170, 172, 173, 175, 176, 178, 181, 182, 184, 186, 187, 189, 191, 192, 195, 196, 202, 203, 204, 205, 207, 209, 211, 214, 215, 216, 217, 218, 219, 220, 221, 222, 225, 226, 228, 229, 230, 236, 238, 240, 241, 242, 243, 247, 249, 250, 251, 253, 255, 256, 257, 259, 262, 264, 265, 266, 271, 276, 278, 279, 280, 282, 286, 287, 288, 289, 290, 291, 293, 297, 298, 299, 301, 303, 310, 311, 312, 313, 315, 316, 317, 318, 320, 322, 323, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 338, 341, 342, 343, 344, 345, 346, 349, 350, 352, 354, 357, 358, 359, 360, 363, 364, 365, 366, 367, 368, 369, 372, 373, 375, 378, 379, 380, 382, 384, 385, 388, 389, 390, 391, 393, 394, 395, 396, 400, 401, 402, 403, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 421, 423, 424, 425, 426, 429, 430, 433, 434, 436, 439, 441, 442, 443, 444, 445, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 461, 462, 464, 465, 468, 471, 472, 473, 474, 477, 478, 479, 482, 485, 486, 490, 494, 495, 496, 497, 498, 500, 501, 504, 505, 506, 507, 511, 512, 513, 514, 515, 516, 517, 518, 519, 521, 523, 525, 526, 527, 528, 529, 530, 532, 533, 534, 535, 536, 538, 539, 540, 541, 542, 543, 545, 546, 547, 548, 549, 550, 552, 553, 554, 556, 557, 558, 559, 560, 561, 562, 563, 565, 568, 571, 572, 573, 575, 576, 578, 580, 582, 583, 586, 589, 590, 591, 594, 596, 597, 598, 600, 601, 603, 604, 605, 606, 607, 609, 610, 611, 612, 613, 614, 615, 616, 617, 619, 620, 622, 623, 624, 625, 627, 628, 629, 632, 633, 635, 636, 637, 638, 641, 645, 646, 647, 649, 650, 652, 654, 657, 658, 660, 661, 663, 664, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 681, 682, 683, 684, 685, 687, 688, 689, 691, 692, 693, 695, 696, 697, 698, 699, 700, 701, 704, 705, 706, 707, 709, 710, 711, 712, 713, 715, 717, 718, 719, 720, 721, 722, 723, 725, 726, 727, 728, 729, 730, 731, 732, 733, 735, 736, 737, 739, 740, 741, 742, 743, 744, 745, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 759, 762, 763, 764, 765, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 778, 779, 780, 781, 783, 784, 785, 786, 787, 788, 789, 791, 792, 793, 794, 795, 796, 797, 798, 799, 801, 803, 804, 805, 806, 807, 808, 809, 810, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 829, 830, 831, 832, 833, 834, 836, 837, 839, 840, 841, 842, 843, 845, 846, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 873, 874, 875, 876, 877, 879, 880, 881, 882, 883, 884, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 932, 933, 934, 935, 937, 938, 939, 940, 941, 942, 943, 944, 945, 947, 948, 949, 950, 951, 952, 953, 954, 956, 957, 958, 959, 960, 961, 962, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 986, 988, 989, 991, 992, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1007, 1008, 1009, 1011, 1012, 1013, 1014, 1015, 1016, 1018, 1019, 1020, 1021, 1022, 1023, 1025, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1036, 1037, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1066, 1067, 1068, 1069, 1070, 1071, 1073, 1074, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112] # for [0.5 0.  0.  0.5 0. ]
        # label = [1, 2, 5, 6, 8, 9, 10, 12, 13, 15, 16, 19, 20, 21, 25, 26, 27, 30, 31, 34, 35, 36, 40, 42, 43, 45, 48, 50, 51, 53, 57, 61, 67, 68, 69, 70, 71, 74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 93, 94, 97, 98, 101, 102, 104, 107, 108, 109, 110, 113, 115, 118, 119, 120, 124, 125, 126, 127, 128, 130, 131, 140, 143, 144, 146, 148, 149, 150, 151, 152, 154, 155, 158, 159, 162, 165, 167, 168, 171, 175, 178, 181, 188, 193, 195, 196, 198, 202, 204, 209, 212, 214, 215, 217, 218, 219, 221, 225, 226, 230, 231, 233, 235, 238, 240, 242, 243, 250, 251, 254, 258, 259, 262, 265, 268, 269, 272, 280, 285, 287, 288, 290, 291, 295, 298, 305, 310, 311, 312, 316, 317, 318, 319, 320, 321, 322, 324, 328, 329, 330, 332, 333, 340, 342, 343, 345, 346, 347, 349, 350, 351, 352, 353, 354, 355, 356, 358, 364, 365, 366, 368, 369, 370, 371, 372, 373, 375, 378, 382, 384, 387, 388, 389, 390, 393, 395, 398, 400, 401, 402, 406, 407, 408, 411, 412, 413, 414, 415, 417, 419, 421, 423, 424, 425, 426, 427, 428, 429, 430, 433, 436, 437, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 451, 452, 455, 456, 457, 463, 464, 465, 468, 471, 472, 473, 474, 476, 477, 479, 481, 482, 484, 489, 490, 494, 496, 497, 498, 500, 501, 503, 506, 507, 509, 512, 513, 514, 515, 516, 517, 518, 522, 523, 525, 528, 529, 530, 532, 534, 535, 536, 538, 539, 540, 542, 545, 546, 548, 549, 550, 551, 554, 556, 558, 559, 561, 563, 565, 567, 568, 570, 571, 572, 573, 577, 578, 579, 580, 581, 583, 584, 588, 589, 598, 599, 602, 603, 604, 606, 607, 610, 612, 613, 615, 618, 619, 620, 622, 623, 625, 626, 627, 629, 630, 632, 633, 636, 637, 638, 640, 646, 647, 648, 649, 650, 651, 653, 656, 657, 659, 661, 663, 669, 670, 671, 672, 674, 676, 677, 682, 684, 685, 689, 690, 691, 692, 693, 695, 698, 701, 703, 704, 705, 706, 708, 714, 717, 720, 721, 723, 724, 728, 729, 731, 732, 733, 735, 738, 741, 743, 744, 749, 753, 755, 756, 757, 761, 763, 767, 769, 770, 771, 773, 774, 775, 776, 782, 789, 790, 791, 794, 796, 798, 802, 804, 806, 807, 808, 809, 814, 825, 826, 827, 830, 831, 836, 842, 843, 846, 848, 851, 852, 853, 856, 860, 861, 864, 865, 866, 867, 868, 869, 872, 879, 880, 881, 887, 888, 889, 891, 892, 893, 897, 900, 902, 903, 904, 907, 910, 911, 912, 914, 916, 918, 919, 920, 923, 926, 929, 930, 932, 933, 936, 939, 940, 944, 945, 947, 948, 952, 953, 954, 957, 958, 960, 963, 965, 967, 969, 971, 972, 973, 974, 976, 977, 978, 984, 988, 994, 996, 997, 998, 999, 1001, 1006, 1007, 1009, 1013, 1014, 1015, 1016, 1019, 1021, 1023, 1025, 1026, 1029, 1037, 1038, 1042, 1043, 1045, 1049, 1050, 1055, 1056, 1057, 1058, 1059, 1060, 1062, 1064, 1066, 1068, 1069, 1076, 1077, 1078, 1079, 1081, 1084, 1085, 1086, 1087, 1090, 1092, 1094, 1096, 1098, 1099, 1105, 1107, 1109, 1110, 663] #[0.5, 0. 0, 0.5, 0] for tsne > 0
        # label = [1, 5, 7, 10, 11, 13, 15, 17, 19, 20, 21, 24, 25, 34, 35, 36, 40, 41, 42, 45, 48, 50, 52, 67, 69, 71, 75, 80, 82, 83, 85, 87, 88, 89, 93, 98, 103, 106, 109, 111, 117, 121, 123, 127, 131, 133, 134, 140, 148, 149, 151, 154, 155, 161, 162, 164, 165, 166, 167, 170, 172, 174, 178, 181, 183, 189, 190, 192, 193, 195, 197, 200, 201, 202, 203, 204, 205, 206, 212, 214, 216, 219, 220, 222, 225, 226, 229, 235, 238, 243, 247, 251, 253, 255, 259, 263, 268, 269, 274, 280, 287, 289, 290, 293, 295, 305, 307, 309, 311, 316, 318, 319, 323, 327, 329, 330, 335, 339, 340, 347, 348, 349, 350, 351, 352, 353, 362, 364, 365, 366, 368, 369, 372, 373, 374, 377, 378, 380, 383, 384, 385, 387, 394, 396, 398, 400, 401, 402, 406, 408, 409, 411, 412, 415, 419, 423, 424, 428, 429, 430, 434, 436, 437, 443, 446, 447, 448, 450, 453, 458, 462, 465, 466, 469, 470, 471, 473, 476, 481, 485, 488, 490, 491, 493, 496, 497, 498, 500, 505, 512, 513, 514, 516, 517, 518, 524, 525, 526, 528, 532, 535, 537, 539, 542, 543, 545, 546, 547, 550, 551, 553, 554, 556, 558, 567, 570, 571, 572, 575, 581, 583, 585, 587, 588, 589, 591, 596, 603, 604, 606, 607, 609, 613, 614, 615, 616, 617, 620, 623, 625, 630, 632, 633, 638, 639, 641, 644, 645, 647, 649, 650, 652, 653, 655, 656, 658, 660, 661, 664, 665, 669, 670, 672, 674, 676, 677, 678, 680, 681, 682, 684, 686, 687, 688, 689, 690, 692, 694, 695, 697, 698, 700, 701, 702, 703, 705, 706, 707, 710, 711, 712, 718, 720, 721, 724, 726, 728, 729, 731, 733, 738, 739, 740, 742, 743, 744, 745, 747, 748, 751, 752, 753, 754, 757, 758, 759, 761, 763, 764, 765, 766, 768, 769, 771, 774, 778, 779, 782, 783, 784, 786, 789, 790, 791, 795, 797, 798, 803, 804, 806, 807, 808, 814, 815, 816, 817, 818, 820, 824, 829, 830, 831, 834, 840, 841, 842, 843, 844, 845, 846, 849, 851, 852, 858, 860, 862, 866, 867, 868, 869, 870, 871, 876, 878, 879, 881, 885, 886, 890, 892, 893, 894, 895, 896, 898, 899, 900, 901, 903, 905, 906, 909, 910, 913, 920, 921, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 939, 941, 942, 945, 946, 949, 950, 952, 953, 956, 957, 958, 959, 962, 963, 964, 966, 969, 970, 972, 974, 975, 976, 977, 978, 980, 981, 983, 984, 985, 986, 989, 991, 992, 993, 995, 997, 999, 1000, 1001, 1002, 1006, 1009, 1011, 1012, 1013, 1014, 1022, 1023, 1024, 1025, 1027, 1030, 1031, 1033, 1037, 1040, 1041, 1042, 1043, 1044, 1046, 1054, 1055, 1057, 1058, 1060, 1061, 1064, 1065, 1067, 1068, 1069, 1075, 1077, 1078, 1079, 1080, 1081, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1093, 1095, 1096, 1098, 1099, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1110, 1111, 1112, 510
# ] [0, 0.5, 0, 0.5, 0] pos > 0
#         label = [0, 3, 5, 6, 11, 17, 18, 20, 26, 27, 28, 29, 31, 32, 37, 38, 46, 47, 49, 53, 55, 59, 60, 62, 64, 65, 72, 73, 75, 77, 78, 84, 85, 90, 91, 94, 95, 96, 99, 107, 108, 116, 120, 121, 122, 123, 128, 130, 132, 138, 141, 142, 143, 145, 147, 148, 149, 153, 154, 156, 159, 162, 164, 169, 171, 173, 175, 179, 181, 182, 184, 185, 186, 187, 188, 190, 199, 203, 207, 209, 210, 212, 223, 224, 227, 228, 231, 233, 234, 235, 237, 238, 239, 240, 248, 249, 251, 252, 254, 258, 260, 263, 265, 266, 270, 274, 275, 276, 277, 278, 280, 282, 284, 285, 287, 289, 290, 296, 299, 321, 324, 325, 333, 336, 337, 342, 343, 347, 350, 354, 356, 357, 358, 359, 360, 361, 362, 363, 365, 367, 371, 380, 386, 391, 392, 393, 396, 397, 399, 403, 405, 407, 409, 410, 412, 417, 418, 431, 432, 435, 438, 440, 441, 450, 451, 457, 459, 460, 467, 468, 470, 472, 479, 484, 486, 488, 492, 493, 496, 497, 498, 499, 503, 504, 508, 509, 513, 520, 522, 531, 533, 537, 541, 542, 544, 547, 550, 552, 559, 560, 561, 562, 567, 569, 574, 575, 577, 579, 580, 582, 584, 586, 592, 593, 594, 597, 600, 606, 608, 618, 624, 626, 631, 634, 637, 640, 645, 649, 654, 661, 665, 666, 667, 669, 673, 676, 678, 682, 684, 686, 692, 694, 696, 699, 700, 701, 702, 705, 706, 707, 709, 711, 712, 713, 714, 718, 719, 725, 726, 729, 730, 733, 734, 736, 739, 740, 742, 743, 746, 747, 748, 749, 751, 752, 755, 757, 758, 760, 762, 763, 764, 765, 766, 768, 771, 774, 777, 778, 779, 780, 786, 787, 788, 789, 790, 793, 794, 795, 797, 798, 799, 800, 801, 802, 805, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 818, 821, 823, 824, 829, 830, 831, 833, 834, 835, 839, 842, 846, 847, 849, 850, 851, 855, 858, 862, 863, 864, 866, 867, 872, 873, 875, 876, 878, 879, 884, 886, 888, 890, 892, 894, 895, 899, 905, 906, 909, 915, 920, 923, 926, 927, 928, 930, 931, 933, 935, 936, 937, 938, 942, 947, 949, 950, 955, 958, 959, 961, 964, 968, 969, 972, 975, 976, 979, 980, 982, 983, 985, 988, 989, 990, 991, 993, 996, 997, 1000, 1002, 1005, 1006, 1008, 1010, 1011, 1012, 1013, 1014, 1016, 1018, 1022, 1024, 1025, 1026, 1029, 1030, 1033, 1034, 1036, 1039, 1040, 1041, 1042, 1044, 1046, 1049, 1051, 1052, 1053, 1054, 1055, 1057, 1058, 1061, 1064, 1065, 1068, 1070, 1072, 1073, 1074, 1077, 1078, 1079, 1080, 1081, 1084, 1086, 1087, 1088, 1091, 1093, 1095, 1096, 1100, 1101, 1102, 1104, 1105, 1106, 1108, 1111, 1112, 510
#         ] # [0, 0.5, 0, 0.5, 0] abs(pos) >0.1
        label = [5, 11, 13, 15, 17, 20, 21, 24, 45, 50, 67, 69, 75, 80, 85, 87, 93, 103, 121, 123, 127, 148, 149, 154, 161, 162, 164, 165, 167, 170, 172, 174, 178, 181, 189, 190, 192, 200, 201, 202, 203, 204, 205, 212, 220, 225, 226, 229, 235, 238, 247, 251, 255, 263, 268, 274, 280, 287, 289, 290, 305, 311, 329, 340, 347, 350, 351, 352, 362, 365, 366, 377, 378, 380, 384, 394, 396, 400, 401, 402, 408, 409, 412, 415, 419, 424, 429, 443, 446, 450, 453, 462, 466, 470, 473, 476, 481, 485, 488, 490, 493, 496, 497, 498, 505, 512, 513, 516, 518, 526, 528, 532, 537, 539, 542, 543, 546, 547, 550, 554, 567, 571, 572, 575, 591, 606, 609, 615, 620, 625, 630, 632, 645, 649, 653, 655, 656, 661, 665, 669, 672, 674, 676, 678, 682, 684, 686, 687, 692, 694, 697, 700, 701, 702, 705, 706, 707, 710, 711, 712, 718, 726, 729, 731, 733, 738, 739, 740, 742, 743, 747, 748, 751, 752, 757, 758, 763, 764, 765, 766, 768, 771, 774, 778, 779, 783, 784, 786, 789, 790, 791, 795, 797, 798, 803, 807, 808, 814, 815, 816, 818, 820, 824, 829, 830, 831, 834, 841, 842, 844, 846, 849, 851, 858, 862, 866, 867, 868, 876, 878, 879, 881, 885, 886, 890, 892, 893, 894, 895, 899, 905, 906, 909, 910, 920, 921, 926, 927, 928, 929, 930, 931, 932, 933, 935, 939, 941, 942, 946, 949, 950, 956, 958, 959, 962, 963, 964, 969, 970, 972, 974, 975, 976, 978, 980, 981, 983, 984, 985, 989, 991, 993, 995, 997, 999, 1000, 1001, 1002, 1006, 1011, 1012, 1013, 1014, 1022, 1024, 1025, 1030, 1033, 1037, 1040, 1041, 1042, 1044, 1046, 1054, 1055, 1057, 1058, 1061, 1064, 1065, 1067, 1068, 1069, 1075, 1077, 1078, 1079, 1080, 1081, 1084, 1086, 1087, 1088, 1093, 1095, 1096, 1101, 1102, 1103, 1104, 1105, 1106, 1108, 1110, 1111, 1112
                ]  #[0, 0.5, 0, 0.5, 0] pos > 0
    return label
def change_label_(y, new_label):
    y_old = [1] * len(y)
    for idx in new_label:
        y_old[idx] = -1
    return np.array(y_old).reshape(len(y),)

def change_label(graph, Y, change_flag='no'):
    if change_flag == 'no':
        return Y
    elif change_flag == 'yes':
        if (graph == 'protein_data') or (graph == 'mutag'):
            Y = change_label_(Y, new_label(graph))
            return Y
def check_kernel():

    pass
def graphsimport():
    import os
    import random;
    np.set_printoptions(precision=4)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    from sklearn import manifold
    import timeit
def embedding_plot(pos, tsne_pos, title_cache, y_color, y, annotate):
    pos = np.array(pos)
    y_color = np.array(y_color)
    assert np.shape(pos)[1]==2
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(211)
    print('Type of pos is %s. Shape of pos is %s'%(type(pos), np.shape(pos)))
    # ax.scatter(pos, c=y_color, s=2)
    ax.scatter(np.array(pos[:, 0]), np.array(pos[:, 1]), c=y_color, s =2) # s = 2
    (beta, graph, round, train_acc, test_acc,c, sigma) = title_cache
    from textwrap import wrap
    title_text = str(beta) + ' ' + graph + ' Round ' + str(round) + 'SVM:Train:' + str(train_acc) + ' Test:' + str(
        test_acc) + ' C: ' + str(c) + ' Sigma: ' + str(sigma)
    title = ax.set_title("\n".join(wrap(title_text, 60)))
    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)

    import matplotlib.pyplot as plt
    for xi, yi, pidi in zip(pos[:, 0], pos[:, 1], range(len(y))):
        if (annotate == 'yes') and (pidi % 30 == 0):
            plt.annotate(str(pidi), xy=(xi, yi), xytext=(xi, yi))  # xytext=(xi+0.05 * np.random.rand(), yi+0.05 * np.random.rand())

    ax = fig.add_subplot(212)
    ax.scatter(tsne_pos[:, 0], tsne_pos[:, 1], c=y_color, s=1.2)
    for xi, yi, pidi in zip(tsne_pos[:, 0], tsne_pos[:, 1], range(len(y))):
        if (annotate == 'yes') and (pidi % 30 == 0):
            plt.annotate(str(pidi), xy=(xi, yi), xytext=(xi, yi))  # xytext=(xi+0.05 * np.random.rand(), yi+0.05 * np.random.rand())
    return fig




def MDS(dist_matrix, y, cache,rs=42, annotate='no', print_flag='False', gd='False'):
    # input is the distance matrix
    # ouput: draw the mds/tsne 2D embedding

    graphsimport()
    np.set_printoptions(precision=3)
    random.seed(rs)
    assert np.shape(dist_matrix)[0] == np.shape(dist_matrix)[1] == len(y)

    (beta, graph, round, train_acc, test_acc, c, sigma) = cache
    train_acc = (100 * train_acc).round()
    test_acc = (100 * test_acc).round()
    cache = (beta, graph, round, train_acc, test_acc, c, sigma)

    start_time = timeit.default_timer()
    mds = manifold.MDS(dissimilarity='precomputed', n_jobs=-1, random_state=rs, verbose=0)
    tsne = manifold.TSNE(metric='precomputed', verbose=0, random_state=rs)
    pos = mds.fit_transform(dist_matrix)
    tsne_pos = tsne.fit_transform(dist_matrix)
    print('Computing MDS and SNE takes %s'%(timeit.default_timer() - start_time))

    print('Right part:')
    for i in range(np.shape(pos)[0]):
        if ((pos[i,0] > 0.0) ) and (print_flag=='True'):
            print('%s,'%i),

    # print(np.shape(pos))
    # y = change_label(y, new_label('mutag'))
    # print(list(y).count(1))
    y_color = [color_map(i) for i in y]
    assert len(y_color) == len(y)
    print(y_color.count('r'))
    assert np.shape(pos)[0] == len(y); assert np.shape(tsne_pos)[0] == len(y)

    fig = embedding_plot(pos, tsne_pos, cache, y_color, y, annotate)

    direct = './Viz_algorithm/' + graph + '/distance_matrix/'; make_direct(direct)
    if gd == 'True':
        direct = './Viz_algorithm/' + graph + '/gd/'; make_direct(direct)
    filename = str(round) + '_mds_' + str(beta) + '.png'
    fig.savefig(direct + filename); print('Saving figure successfully')


# SVM realted
def compute_alpha_kernel(kernel, Y, lbd):
    import numpy as np
    n = np.shape(Y)[0]
    inv_K = np.linalg.inv(kernel + lbd * np.eye(n))
    alpha = np.dot(inv_K, Y)
    assert len(alpha) == n
    return alpha
class kernelMachine():
    def __init__(self, kernel, c, Y, lbd = 1e-7, loss='hinge', debug='off'):
        self.kernel = kernel / np.linalg.norm(kernel, 2)
        self.c = c
        self.n = np.shape(Y)[0]
        self.Y = Y.reshape(self.n,1)
        self.lbd = lbd
        self.loss = loss
        self.debug= debug
        self.test_accuracies = []
        self.train_accuracies = []

    def compute_alpha_kernel(self):
        import numpy as np
        n = np.shape(self.Y)[0]
        inv_K = np.linalg.inv(self.kernel + self.lbd * np.eye(n))
        alpha = np.dot(inv_K, self.Y)
        assert len(alpha) == n
        return alpha

    def get_alpha(self, clf, n):
        # get the dual coefficient
        # X = np.array([[-1, -1], [-12, -2], [1, 1], [2, 1], [4, 8]])
        #         y = np.array([0, 0, 1, 1, 1])
        #         from sklearn.svm import SVC, SVR
        #         clf = SVC(gamma=1)
        #         clf.fit(X, y)
        #
        alpha = np.zeros(n)
        sprt_idx = clf.support_
        dual_coef = clf.dual_coef_
        k = 0
        for i in sprt_idx:
            alpha[i]= dual_coef[0][k] # need to change if want to handle multiple classes
            k = k + 1
        if self.debug=='on':
            print(dual_coef)
            print(clf.n_support_)
            print(k, np.sum(clf.n_support_))
        # assert k ==np.sum(clf.n_support_)
        assert  k == len(clf.support_)
        return alpha

    def single_test(self):
        from sklearn import svm
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import accuracy_score

        loss = self.loss
        kernel = self.kernel
        n = self.n
        Y = self.Y
        c = self.c

        print('.'),
        y_train, y_test, indices_train, indices_test = train_test_split(Y, range(n), test_size=0.1, random_state=i)
        n_train = len(y_train)
        # kernel = np.exp(np.multiply(dist_matrix, -dist_matrix) / sigma)
        kernel_train = kernel[np.ix_(indices_train, indices_train)]
        assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train) == True
        kernel_test = kernel[np.ix_(indices_test, indices_train)]
        np.shape(kernel_test)

        if loss == 'hinge':
            clf = svm.SVC(kernel='precomputed', C=c)
            clf.fit(kernel_train, y_train)
            y_pred = clf.predict(kernel_test)
            y_reg = y_pred # need to change later
            alpha = self.get_alpha(clf, n_train)

            y_pred_train = clf.predict(kernel_train)
            # print('Train accuracy is %s'%accuracy_score(y_train, y_pred_train))

        elif loss == 'square':
            lbd = self.lbd
            alpha = self.compute_alpha_kernel(kernel_train, y_train, lbd)
            K = kernel_test
            assert np.shape(K)[1] == len(alpha)
            y_pred = np.dot(kernel_test, alpha)
            if self.debug=='on':
                print('The regression in belkin model is %s'%y_pred)
            y_reg = y_pred # get the r_reg for gradient descent
            y_pred = np.sign(y_pred)
            # y_pred = ((y_pred + 1) / 2).astype(int)

            y_pred_train = np.sign(np.dot(kernel_train, alpha))
            # print('Train accuracy is %s' % accuracy_score(y_train, y_pred_train))

        elif loss == 'svr':
            clf = svm.SVR(kernel='precomputed', C=c)
            clf.fit(kernel_train, y_train)
            y_pred = clf.predict(kernel_test)
            y_reg = y_pred
            y_pred = np.sign(y_pred)
            alpha = self.get_alpha(clf, n_train)

            y_pred_train = np.sign(clf.predict(kernel_train))
            # print('Train accuracy is %s' % accuracy_score(y_train, y_pred_train))

        self.test_accuracies.append(accuracy_score(y_test, y_pred))
        self.train_accuracies.append(accuracy_score(y_train, y_pred_train))
        assert 0 not in self.test_accuracies
        assert 0 not in self.train_accuracies

    def multi_tests(self):
        for i in range(10):
            self.single_test()
        print('Loss: %s, Training accuracy is %s' % (self.loss, np.mean(self.train_accuracies)))
        if (self.loss == 'hinge') or (self.loss == 'svr'):
            print('c is %s, mean accuracy is %s, std is %s \n' % (self.c, np.mean(self.test_accuracies), np.std(self.test_accuracies)))
        elif self.loss == 'square':
            print('lambda is %s, mean accuracy is %s, std is %s \n' % (
            self.lbd, np.mean(self.test_accuracies), np.std(self.test_accuracies)))

        # return {'reg': self.y_reg, 'test_idx': self.indices_test, 'train_idx': indices_train, 'coef': alpha}
# km = kernelMachine(kernel=kernel, c=1, Y=Y)

# import dionysus as d
# import numpy as np
# diag1 = np.array([[1,2],[2,3],[6,8]])
# diag2 = np.array([[1.1,2.8],[2.2,3.9],[6.1,20]])
# diag1 = [[1,2],[2,3],[6,8]]
# diag2 = [[1.1,2.8],[2.2,3.9],[6.1,20]]
#
# diag1 = random_lists(50)
# diag2 = random_lists(20)
# dgm1 = d.Diagram([(p[0], p[1]) for p in diag1]) # convert a list of lists to a list of tuples
# dgm2 = d.Diagram([(p[0], p[1]) for p in diag2])
# bd = d.bottleneck_distance(dgm1, dgm2, delta=0.001)
# print d.bottleneck_distance_with_edge(dgm1, dgm2, delta=0.001)
# d_bottleneck_distance_with_edge_base(diag1, diag2, bd=bd)
#
#
def padding_zero(alpha, n, support):
    # take dual coefficeitn and pad zero
    # input: np.array; expect clf.support_ that gives the location of nonzero coefficient
    assert np.shape(alpha)[1] == len(support)
    lis = [0] * n
    for i in range(len(support)):
        idx = support[i]
        # print('right is %s'% alpha[0][i])
        lis[idx] = alpha[0][i]
    lis = np.array(lis)
    assert len(lis) == n
    return lis

##############################

####################################################
# Homology computation
def find_cycle(g, node, debug='on'):
    from networkx import cycle_basis
    basis_list = cycle_basis(g)
    print('Number of basis is %s' % len(basis_list))
    homology_list = []
    # print('New node is %s'%node[0])
    # print('Basis list is %s'%basis_list)
    for lis in basis_list:
        if node[0] in lis:
            birth = birth_comp(lis, g, sub=True)
            death = birth_comp(lis, g, sub=False)
            # print(lis),
            # print(birth),
            # print(death)
            homology_list.append({'birth': birth, 'death': death, 'potential_good_basis': lis})
        else:
            homology_list.append({'birth': None, 'death': None, 'bad_basis': lis})
            if debug == 'on':
                print('The old cycle is %s' % lis)

    if homology_list != []:
        return homology_list
def node_filter(g, threshold, sub=True):
    # output node values below/above certain threshold
    nodesless = []
    if sub == True:
        for (n, d) in g.nodes(data=True):
            if d['fv_random'] <= threshold:
                assert g.node[n]['fv_random'] <= threshold
                nodesless.append(n)
    elif sub == False:
        for (n, d) in g.nodes(data=True):
            if d['fv_random'] >= threshold:
                assert g.node[n]['fv_random'] >= threshold
                nodesless.append(n)

    return nodesless
def cycle_plus(lis1, lis2):
    # needs refinment
    # make sure the common insersection in lis1 is increasing
    import numpy as np
    assert type(lis1) == type(lis2) == list
    assert len(lis1) == len(np.unique(lis1))
    assert len(lis2) == len(np.unique(lis2))

    common = list(set(lis1) & set(lis2))
    idx1 = [lis1.index(i) for i in common]
    idx2 = [lis2.index(i) for i in common]
    idx1_diff = [t - s for s, t in zip(idx1, idx1[1:])]
    if -1 in idx1_diff:
        lis1.reverse()
        idx1 = [lis1.index(i) for i in common]
        idx1_diff = [t - s for s, t in zip(idx1, idx1[1:])]
    assert -1 not in idx1_diff
    idx2_diff = [t - s for s, t in zip(idx2, idx2[1:])]
    assert len(idx1_diff) == len(idx2_diff)
    new_cycle = []
    if idx1_diff == idx2_diff == [1] * len(idx1_diff):
        i = idx1[-1]
        while i % len(lis1) != idx1[0]:
            new_cycle.append(lis1[i % len(lis1)])
            i = i + 1

        i = idx2[0]
        while i % len(lis2) != idx2[-1]:
            new_cycle.append(lis2[i % len(lis2)])
            i = i - 1

    elif idx1_diff == [-i for i in idx2_diff] == [1] * len(idx1_diff):
        i = idx1[-1]
        while i % len(lis1) != idx1[0]:
            new_cycle.append(lis1[i % len(lis1)])
            i = i + 1

        i = idx2[0]
        while i % len(lis2) != idx2[-1]:
            new_cycle.append(lis2[i % len(lis2)])
            i = i + 1
    else:
        raise AssertionError
    return new_cycle
# lis1 = [19, 20, 21, 22, 3, 4]
# lis1.reverse()
# lis2 = [1, 2, 3, 4, 5, 6, 15, 14, 13, 12, 11, 10, 9, 0]
# cycle_plus(lis1, lis2)
def merge_two_dicts(x, y):
    assert type(x) == type(y) == dict
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z
def get_pers_data(g, fv_list, order='increasing'):
    pers_data = list()
    if order == 'increasing':
        rangelist = range(len(fv_list))
    elif order == 'decreasing':
        rangelist = range(len(fv_list), -1, -1)

    for i in rangelist:
        import networkx as nx
        threshold = fv_list[i]
        if i == 0:
            diff = set(node_filter(g, threshold, False))
            newnode = list(diff)
        elif i > 0:
            old_threshold = fv_list[i - 1]
            diff = set(node_filter(g, threshold, False)) - set(node_filter(g, old_threshold, False))
            newnode = list(diff)
            print('Threshold is %s, New nodes are %s' % (threshold, newnode))

        g_sub = g.subgraph(node_filter(g, threshold, False))
        component = [list(i) for i in list(sorted(nx.connected_components(g_sub)))]
        print('Number of component is %s, The component lists are: %s' % (len(component), component)),
        print('\n')
        homology1_list = find_cycle(g_sub, newnode)
        homology0_list = [{'Threshold': threshold, 'Node': list(diff), 'N_Comp': len(component), 'Comp': component}]
        pers_data = pers_data + [homology0_list]
        # pers_data = pers_data + [homology0_list, homology1_list]
        print(g_sub.nodes())
        print(g_sub.edges())

    return pers_data
def diff(first, second):
    # first: set of sets
    first = set(frozenset(i) for i in first)
    second = set(frozenset(i) for i in second)
    second = set(second)
    return {'d2': [list(item) for item in second if item not in first],
            'd1': [list(item) for item in first if item not in second]}
def birth_comp(lis, g, sub=True):
    if (sub == True) or (sub == 'sub') or (sub == 'min'):
        birth = 1e5
        for i in lis:
            if birth > g.node[i]['fv']:
                birth = g.node[i]['fv']
        return birth
    elif (sub == False) or (sub == 'super') or (sub == 'max'):
        birth = -1e5
        for i in lis:
            if birth < g.node[i]['fv']:
                birth = g.node[i]['fv']
        return birth
def zero_homology(pers_data):
    for i in range(len(pers_data) - 1):
        print('\n')
        difference = diff(pers_data[i][0]['Comp'], pers_data[i + 1][0]['Comp'])
        print(difference),
        print(pers_data[i + 1][0]['Node'])
        newnode = pers_data[i + 1][0]['Node'][0]
        b_list = difference['d1']
        a_list = difference['d2']

        if len(b_list) == 0:
            assert len(a_list) == 1
            assert len(a_list[0]) == 1
            # assert b_list[0] == pers_data[i+1]['Node']
            print('New node is a new component: Node %s, value %s ' % (
            pers_data[i][0]['Node'], pers_data[i][0]['Threshold']))
            # print('The birth time of new node %s is %s'%(newnode ,g.node[newnode]['fv_random']))
        elif len(b_list) == 1:
            if a_list[0].sort() == list((b_list[0] + pers_data[i + 1][0]['Node'])).sort():
                print('New node merged in the existing component')

        elif len(b_list) == 2:
            if a_list[0].sort() == list((b_list[0] + b_list[1] + pers_data[i + 1][0]['Node'])).sort():
                print(('New node connect two existing components %s and %s' % (b_list[0], b_list[1])))
                if birth_comp(b_list[0], g, False) < birth_comp(b_list[1], g, False):
                    print('The birth time of %s is %s' % (b_list[0], birth_comp(b_list[0], g, 'max')))
                    print('The death time of %s is %s' % (b_list[0], g.node[newnode]['fv_random']))

                elif birth_comp(b_list[0], g, False) >= birth_comp(b_list[1], g, False):
                    print('The birth time of %s is %s' % (b_list[1], birth_comp(b_list[1], g, 'max')))
                    print('The death time of %s is %s' % (b_list[1], g.node[newnode]['fv_random']))

                else:
                    print('Unconsidered cases')
                    raise Exception

        elif len(b_list) == 3:
            print('One node kills more than one component')
            raise AssertionError
        else:
            print('Unconsiderd cases')
def get_matrix(dgms, dtype='bottleneck'):
    # get dist matrix from dgms(a list of pds)
    # serial computing
    import numpy as np
    import dionysus as d
    assert type(dgms) == list
    n = len(dgms)
    dist_matrix = np.zeros((n, n))+ 1.111
    idx_dict = {}
    if dtype == 'bottleneck':
        for i in range(n):
            for j in range(n):
                dist = d.bottleneck_distance_with_edge(dgms[i], dgms[j])
                print(dist)
                dist_matrix[i, j] = dist[0]
                (idx1, idx2) = dist[1]
                if ((str(idx1)=='-1') and (str(idx2)=='-1')):
                    idx_dict[(i, j)] = 'same'
                elif ((str(idx1)!='-1') and (str(idx2)!='-1')):
                    idx_dict[(i, j)] = ((dgms[i][idx1].birth, dgms[i][idx1].death), (dgms[j][idx2].birth, dgms[j][idx2].death))
                elif ((str(idx1)!='-1') and (str(idx2)=='-1')) :
                    idx_dict[(i, j)] = ((dgms[i][idx1].birth, dgms[i][idx1].death), None)
                elif ((str(idx1)=='-1') and (str(idx2)!='-1')):
                    idx_dict[(i, j)] = (None, (dgms[j][idx2].birth, dgms[j][idx2].death))
                else:
                    raise AssertionError
                    # assert np.amax(abs((dist_matrix - dist_matrix.T))) < 0.02 # db is not symmetric
        return ((dist_matrix + dist_matrix.T) / 2.0, idx_dict)
    elif dtype == 'wasserstein':
        for i in range(n):
            for j in range(n):
                import dionysus as d
                dist = d.wasserstein_distance(dgms[i], dgms[j], q=2)
                dist_matrix[i, j] = dist
        # assert np.amax(abs((dist_matrix - dist_matrix.T))) < 0.02 # db is not symmetric
        return (dist_matrix + dist_matrix.T) / 2.0
# (dist_matrix, idx_dict) = get_matrix(pd_dgms, dtype='bottleneck')

########################
# def get_vector(dgms, i):
#     # dgms here is a list of several tuples
#     import dionysus as d
#     import numpy as np
#     n = len(dgms)
#     dist_matrix = np.zeros((n, n)) + 1.111
#     dist_vec = np.zeros((1, n)) + 1.11
#     idx_dict = {}
#     dgm_i = d.Diagram(dgms[i])
#     for j in range(n):
#         dgm_j = d.Diagram(dgms[j])
#
#         try:
#             dist = d.bottleneck_distance_with_edge(dgm_i, dgm_j)
#             dist_vec[0, j] = dist[0]
#         except AssertionError:
#             print('diagram %s, %s has problem' % (i, j))
#             continue
#
#         idx_dict[(i,j)] = dist[1]
#         (idx1, idx2) = dist[1]
#         if ((str(idx1) == '-1') and (str(idx2) == '-1')):
#             idx_dict[(i, j)] = 'same'
#         elif ((str(idx1) != '-1') and (str(idx2) != '-1')):
#             idx_dict[(i, j)] = (
#                 (dgms[i][idx1][0], dgms[i][idx1][1]), (dgms[j][idx2][0], dgms[j][idx2][1]))
#         elif ((str(idx1) != '-1') and (str(idx2) == '-1')):
#             idx_dict[(i, j)] = ((dgms[i][idx1][0], dgms[i][idx1][1]), None)
#         elif ((str(idx1) == '-1') and (str(idx2) != '-1')):
#             idx_dict[(i, j)] = (None, (dgms[j][idx2][0], dgms[j][idx2][1]))
#         else:
#             raise AssertionError
#             # assert np.amax(abs((dist_matrix - dist_matrix.T))) < 0.02 # db is not symmetric
#     assert np.shape(dist_vec) == (1, n)
#     return dist_vec
#
# # serial
# vec_serial = []
# n = len(dgms)
# for i in range(n):
#     vec_serial.append(get_vector(dgms, i))
#
# # parallel
# vec_parallel = Parallel(n_jobs=-1)(delayed(get_vector)(dgms, i) for i in range(n))
#
# # test serial=parallel
# for i in range(n):
#     try:
#         assert (vec_serial[i] == vec_parallel[i]).all()
#     except AssertionError:
#         print('Line %s has problem' % i)
#
# tuple_dgms = dgms
# # serial
# pd_dgms = [d.Diagram(i) for i in tuple_dgms]
##################

'''
    some test code
    for j in range(len(data)):
        for i in range(len(data)):
            print(j,i),
            # i = 20; j = 12
            bd = d.bottleneck_distance(dgms[i], dgms[j])
            d_bottleneck_distance_with_edge(dgms[i], dgms[j], bd)

    for i in range(len(data)):
        get_matrix_i(i, dgms)

    for i in range(len(dgms)):
        print('Length is %s, %s diagram'%(len(dgms[i]),i)),
        print_dgm(dgms[i])
        print('\n')

    68 diagram (0.166667,0.277778), (0.166667,0.277778), (0.25,0.277778), (0.25,0.277778), (0.25,0.277778), (0.25,0.277778), (0.25,0.277778), (0.25,0.277778), (0.5,0.611111), (0.5,0.611111), (0.166667,0.666677), (0.625,1), (0.625,1), (0.625,1), (0.625,1.00001)
    95 diagram (-4.81688e-11,0.0416667), (-4.81685e-11,0.0416667), (-4.0862e-11,0.0416667), (-4.08615e-11,0.0416667), (0.5,0.625), (0.5,0.625), (0.5,0.625), (0.5,0.625), (0.5,0.625), (0.5,0.625), (-0.0416667,0.62501)
    #
    #
    for i in range(188):
        for j in range(188):
                try:
                    print (i, j, d.bottleneck_distance(dgms[i],dgms[j]))
                    print (i, j, d_bottleneck_distance_with_edge(dgms[i],dgms[j]))
                except RuntimeError:
                    print(i, j, )
    for i in range(188):
        try:
            get_matrix_i(i, dgms)
        except:
            print('Dgms %s has problem'%(i))
    '''

# (K_wl_2, K_wl_2_after, dist) = gk_dist('nci1')
# Y = np.array([-1] * 1649 + [1] * 1646 + [-1] * 403 + [1] * (412))
# MDS_simple(dist, Y, 42)
