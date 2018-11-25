import numpy as np
import sys
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/')
from utilities import *
from condition import *
# FUNCTION_TYPE = sys.argv[1]
from joblib import Parallel,delayed
traing_num = 20
score = 0
tuned_parameters = TUNED_PARAMETERS

def call_i_example(i, type, filtration, theta=15):
    ## here filtration only takes sub or super
    def vectorize(id=3, type='ricci', hom=0, filtration='sub'):
        result = read_data(id, type, hom, filtration, theta=theta)
        data = result['data']; imax = result['max']; imin = result['min']
        if filtration=='super':
            data = rotate_data(data,super_check=True)
        elif filtration=='sub':
            data = rotate_data(data,super_check=False)

        vector = draw_data(data,imax,imin)
        vector = np.array(vector).reshape(1, len(vector))
        return vector

    v1 = vectorize(i, type, 0, filtration)
    v2 = vectorize(i, type, 1, filtration)
    v = np.concatenate((v1, v2), axis=1); V = 0;
    if HOMOLOGY_TYPE=='0':
        V = v1
    elif HOMOLOGY_TYPE=='1':
        V = v2
    elif HOMOLOGY_TYPE=='01':
        V = v
    return V

def compute_vector(type, filtration, theta=15):
    from sklearn.preprocessing import normalize
    X_range = X_RANGE
    pdata = Parallel(n_jobs=-1)(delayed(call_i_example)(i, type, filtration, theta) for i in X_range)
    pdata = np.array(pdata)
    pdata = np.squeeze(pdata)
    print(np.shape(pdata))
    pdata = normalize(pdata, norm='l2', axis=1)
    return pdata

def check():
    for i in range(1,10,11920):
        try:
            pass
            # call_i_example(i,'edge_probability_deg','super')
            # call_i_example(i,'edge_probability_deg','sub')
            # print('graph %s is good'%(i))
        except:
            print('graph %s has not been processed'%(i))
            continue

def laplacian_kernel(X,sigma):
    from scipy.spatial.distance import pdist,squareform
    dist_matrix = squareform(pdist(X,'euclidean'))
    return np.exp(dist_matrix/sigma)

def clf_laplacian(c,sigma,X,Y):
    from sklearn import svm
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score
    n = np.shape(X)[0]
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, range(n), test_size=0.1,
                                                                                     random_state=1)
    svc = svm.SVC(kernel='precomputed', C=c)
    kernel = laplacian_kernel(X, sigma)
    kernel_train = laplacian_kernel(X_train, sigma)
    assert np.array_equal(kernel[np.ix_(indices_train, indices_train)], kernel_train)==True
    svc.fit(kernel_train, y_train)
    kernel_test = kernel[np.ix_(indices_test, indices_train)]
    np.shape(kernel_test)
    y_pred = svc.predict(kernel_test)
    print('c is %s, sigma is %s, accuracy is %s' % (c, sigma, accuracy_score(y_test, y_pred)))

def test_laplacian_kernel():
    from joblib import Parallel, delayed
    clist = [0.01, 0.1, 1, 2, 5, 10, 20, 30]
    sigmalist = [0.01, 0.1, 1, 10, 100]
    Parallel(n_jobs=30)(delayed(clf_laplacian)(c, sigma, X, Y) for c in clist for sigma in sigmalist)

def save():
    pass
    # V = compute_vector('ricci','sub')
    # np.shape(V)
    ### compute kernel and save if
    # kernel = np.dot(vector,vector.T)
    # import scipy.io
    # file_directory ="/home/cai.507/Documents/DeepLearning/deep-persistence/nci1/" + 'linear_kernel.mat'
    # scipy.io.savemat(file_directory,mdict={'kernel':kernel})

def clf_search(X, Y, random_state=1):
    print('Start Grid Search')
    import os
    from sklearn import svm
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report, accuracy_score
    for score in ['accuracy']:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=random_state)
        import time;
        start = time.time()
        if GRAPH == 'reddit_12K':
            from sklearn.svm import LinearSVC
            clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='%s' % score, n_jobs=-1, verbose=1)
        else:
            clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, scoring='%s' % score, n_jobs=-1, verbose=1)
        clf.fit(X_train, y_train)
        end = time.time();
        print('the total training time is %s' % (end - start))
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
        print('Accuracy on the test data!')
        print(accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))

        direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH + '/report/' + PARAMETER_CONTROL + '/' #+ PARAMETER_CONTROL + '/'
        if not os.path.exists(direct):
            os.makedirs(direct)
        import datetime
        now = datetime.datetime.now()
        file_name = FUNCTION_TYPE + '-' + HOMOLOGY_TYPE + '-' + FILTRATION_TYPE + '-' + TEST + '.txt'
        file = open(direct + file_name,'a')
        file.write('Date and Time is %s' %(now))
        file.write('background info: \n  GRAPH is %s \n Test is %s \n Parameters are %s\n Filtration Function is %s \n Filtration Type is %s\n\n'%(GRAPH, TEST,TUNED_PARAMETERS, FUNCTION_TYPE, FILTRATION_TYPE))
        file.write('the total training time is %s \n' % (end - start))
        file.write("Best parameters set found on development set is \n %s with score %s\n\n" % (clf.best_params_, clf.best_score_))
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            file.write("%0.3f (+/-%0.03f) for %r \n"% (mean, std * 2, params))
        file.write('the accuracy on the test test is %s'%accuracy_score(y_true, y_pred))
        file.write(classification_report(y_true, y_pred))
        file.close()
        return {'param': clf.best_params_, 'score': clf.best_score_}

def clf_search_offprint(X, Y, random_state=1):
    import os
    from sklearn import svm
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report, accuracy_score
    for score in ['accuracy']:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=random_state)
        # clf = svm.SVC(kernel='linear', C=6)
        # clf = AdaBoostClassifier(svm.SVC(probability=True, kernel='linear'))
        # import time
        # from sklearn.multiclass import OneVsRestClassifier
        # from sklearn.ensemble import BaggingClassifier
        # start = time.time()n
        # clf = OneVsRestClassifier(BaggingClassifier(svm.SVC(kernel='linear', probability=True,class_weight='auto'),max_samples=1/20, n_estimators=10))
        # clf.fit(X_train,y_train)
        # end = time.ime()
        # print(end-start, clf.score(X_train,y_train))
        # s = clf.score(X_test,y_test)
        # score = score + s/traing_num
        import time;
        start = time.time()
        if GRAPH == 'reddit_12K':
            from sklearn.svm import LinearSVC
            clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='%s' % score, n_jobs=-1, verbose=0)
        else:
            clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, scoring='%s' % score, n_jobs=-1, verbose=0)
        clf.fit(X_train, y_train)
        end = time.time();
        # print('the total training time is %s' % (end - start))
        # print("Best parameters set found on development set is \n %s with score %s" % (
        # clf.best_params_, clf.best_score_))
        # print(clf.best_params_)
        # print("Grid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            pass
            # print("%0.3f (+/-%0.03f) for %r"
            #       % (mean, std * 2, params))
        # print("Detailed classification report:\n")
        # print("The model is trained on the full development set.")
        # print("The scores are computed on the full evaluation set.")
        # print()
        y_true, y_pred = y_test, clf.predict(X_test)
        # print('accuracy first!')
        # print(accuracy_score(y_true, y_pred))
        # print(classification_report(y_true, y_pred))

        direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH + '/report/' + PARAMETER_CONTROL + '/' #+ PARAMETER_CONTROL + '/'
        if not os.path.exists(direct):
            os.makedirs(direct)
        import datetime
        now = datetime.datetime.now()
        file_name = FUNCTION_TYPE + '-' + HOMOLOGY_TYPE + '-' + FILTRATION_TYPE + '-' + TEST + '.txt'
        file = open(direct + file_name,'a')
        file.write('Date and Time is %s' %(now))
        file.write('background info: \n  GRAPH is %s \n Test is %s \n Parameters are %s\n Filtration Function is %s \n Filtration Type is %s\n\n'%(GRAPH, TEST,TUNED_PARAMETERS, FUNCTION_TYPE, FILTRATION_TYPE))
        file.write('the total training time is %s \n' % (end - start))
        file.write("Best parameters set found on development set is \n %s with score %s\n\n" % (clf.best_params_, clf.best_score_))
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            file.write("%0.3f (+/-%0.03f) for %r \n"% (mean, std * 2, params))
        file.write('the accuracy on the test test is %s'%accuracy_score(y_true, y_pred))
        file.write(classification_report(y_true, y_pred))
        file.close()
        return {'param': clf.best_params_, 'score': clf.best_score_}

def evaluate_best_estimator(grid_search_re, X, Y):
    print('Start evaluating the best estimator...')
    import time
    from sklearn import svm

    # create a file to write
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH + '/result/' + PARAMETER_CONTROL + '/'  # + PARAMETER_CONTROL + '/'
    if not os.path.exists(direct):
        os.makedirs(direct)
    file_name = FUNCTION_TYPE + '-' + HOMOLOGY_TYPE + '-' + FILTRATION_TYPE + '-' + TEST + '.txt'
    file = open(direct + file_name, 'a')

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
    now = datetime.datetime.now()
    file.write('%s\n'%str(now))
    # scores = cross_val_score(clf, X, Y, cv=10, n_jobs=-1);
    # print(scores);
    # print("Accuracy: %0.3f (+/- %0.3f) \n" % (scores.mean(), scores.std() * 2))

    # try another one
    cv_score = []
    if (GRAPH == 'reddit_12K') or (GRAPH == 'reddit_5K') or (GRAPH == 'nci1') or (GRAPH == 'nci109'):
        n = 3
    else:
        n = 11
    for i in range(1, n):
        # print('this is cv %s' % i)
        k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        scores = cross_val_score(clf, X, Y, cv=k_fold, scoring='accuracy', n_jobs=-1)
        # print(scores)
        for item in scores:
            file.write('score is %0.3f' %item)
        file.write('\n')
        cv_score.append(scores.mean())
        # print("Accuracy: %0.3f (+/- %0.3f) \n" % (scores.mean(), scores.std() * 2))
        file.write("Accuracy: %0.3f (+/- %0.3f) \n" % (scores.mean(), scores.std() * 2))
    cv_score = np.array(cv_score)
    print('After averageing 10 cross validations, the mean accuracy is %0.3f, the std is %0.3f\n' %(cv_score.mean(), cv_score.std()))
    file.write('After averageing 10 cross validations, the mean accuracy is %0.3f, the std is %0.3f\n' %(cv_score.mean(), cv_score.std()))
    end = time.time()

    # start writing to files
    file.write(
        'background info: \n  GRAPH is %s \n Test is %s \n Parameters are %s\n Filtration Function is %s \n Filtration Type is %s\n\n' % (
            GRAPH, TEST, TUNED_PARAMETERS, FUNCTION_TYPE, FILTRATION_TYPE))
    file.write('the total training time is %s \n' % (end - start))
    file.write(
        "Best parameters found by grid search is %s and the score is %s  " % (grid_search_re['param'], grid_search_re['score']))
    file.write('The end\n\n\n')
    file.close()
    return 0

def rfclf(X,Y):
    import numpy as np
    if np.shape(X)==(1,2):
        return
    print('Try Random Forest, n_estimators=40, max_features=40 '),
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    score1 = np.array([0, 0])
    # score1 = cross_val_score(clf1, X, Y, cv=10)
    # score1 = np.array([0,0])
    clf2 = RandomForestClassifier(n_estimators=40, max_features=40, max_depth=None, min_samples_split=2, random_state=0,n_jobs=-1)
    score2 = cross_val_score(clf2, X, Y, cv=10)

    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH + '/rf/' + PARAMETER_CONTROL + '/'  # + PARAMETER_CONTROL + '/'
    if not os.path.exists(direct):
        os.makedirs(direct)
    file_name = FUNCTION_TYPE + '-' + HOMOLOGY_TYPE + '-' + FILTRATION_TYPE + '-' + TEST + '.txt'
    file = open(direct + file_name, 'a')
    file.write(' use random forest, n_estimators=40, max_features=40, max_depth=None, min_samples_split=2, random_state=0\n')
    file.write('the score is %s'%score2)
    file.write('\n the mean is %s, the std is %s\n'%(score2.mean(), score2.std()))

    file.close()

    return (score1.mean(), score2.mean())

def initializeX(FILTRATION_TYE, FUNCTION_TYPE, theta=15):
    # initialization
    function_type = [FUNCTION_TYPE]
    X_data = {}; Y = {}; Y['data'] = Y_RANGE; typelist = []; X = np.array([[1,1]])
    for typ in function_type:
        X_data[typ] = {}

    # case where FUNCTION_TYPE is not comb
    from sklearn.preprocessing import normalize
    if FUNCTION_TYPE !='comb':
        if FILTRATION_TYPE == 'sub':
            for typ in function_type:
                X = compute_vector(FUNCTION_TYPE, 'sub', theta)
                X = normalize(X, norm='l2', axis=1)

                # X_data[typ]['data'] = compute_vector(typ, 'sub', theta)
                # X_data[typ]['data'] = normalize(X_data[typ]['data'], norm='l2', axis=NORMALIIZATION, copy=True)
            print("Finish computing X(sub). Function type is %s " % function_type)
        elif FILTRATION_TYPE == 'super':
            for typ in function_type:
                X_data[typ]['data'] = compute_vector(typ, 'super',theta)
                X_data[typ]['data'] = normalize(X_data[typ]['data'], norm='l2', axis=NORMALIIZATION, copy=True)
            print("Finish computing X(super). Function type is %s " % function_type)
        elif (FILTRATION_TYPE == 'ss') and (FUNCTION_TYPE != 'edge_prob') and (FUNCTION_TYPE != 'edge_prob_deg'):
            X = np.concatenate((compute_vector(FUNCTION_TYPE, 'sub', theta), compute_vector(FUNCTION_TYPE, 'super', theta)), axis=1);
            # X = compute_vector(FUNCTION_TYPE, 'sub', theta)

            X = normalize(X, norm='l2', axis=1)

    # initialize X
    if FUNCTION_TYPE == 'edge_prob' or FUNCTION_TYPE == 'edge_prob_deg':
        X = np.concatenate(
            (compute_vector(FUNCTION_TYPE + '_sub', 'sub', theta), compute_vector(FUNCTION_TYPE + '_super', 'super', theta)), axis=1);
        X = normalize(X)

    if FUNCTION_TYPE == 'comb':
        # initialize typelist
        # typelist = ['edge_probability', 'ricci_edge', 'ricci', 'deg', 'hop', 'label']
        typelist = ['label', 'laplace', 'laplace1', 'laplace2', 'laplace3', 'laplace4']
        if (GRAPH == 'reddit_5K') or (GRAPH == 'reddit_12K') or (GRAPH == 'reddit_binary'):
            typelist = ['edge_probability', 'ricci_edge', 'ricci', 'deg', 'hop']
            typelist = ['ricci_edge', 'deg']
        if (GRAPH == 'nci1') or (GRAPH =='nci109'): typelist = ['label', 'ricci', 'ricci_edge']
        if (GRAPH == 'enzyme'): typelist = ['ricci', 'deg', 'hop', 'ricci_edge' ]

        # initialize X_data
        for typ in typelist:
            X_data[typ] = {}
            X_data[typ]['data'] = np.array(
                np.concatenate((compute_vector(typ, 'sub', theta), compute_vector(typ, 'super', theta)), axis=1))
            X_data[typ]['data'] = normalize(X_data[typ]['data'], norm='l2', axis=NORMALIIZATION, copy=True)
            print(np.linalg.norm(X_data[typ]['data'], axis=1))
            assert (max(np.linalg.norm(X_data[typ]['data'], axis=1))-1)<0.05

    data = {}; X = normalize(X, norm='l2', axis =1, copy=True)
    data['X_data'] = X_data; data['X']=X; data['typelist'] = typelist
    assert type(data)==dict
    return data

def duplicate(X,n=3):
    # find how many duplicateds in input X
    for i in range(n):
        # np.save('duplicate/' + GRAPH + '_' + FUNCTION_TYPE, X)
        A = X
        x = np.random.rand(A.shape[1])
        y = A.dot(x)
        print('There are %s duplicates in featuere vector' % (len(Y_RANGE) - len(np.unique(y))))
        good_count = 0; bad_count = 0; total_count = 0;
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if (y[i] == y[j]) and (Y_RANGE[i]!=Y_RANGE[j]):
                    bad_count = bad_count + 1
                if (y[i] == y[j]) and (Y_RANGE[i]==Y_RANGE[j]):
                    good_count = good_count + 1
        total_count = good_count + bad_count
        if total_count!=0:
            good_ratio = good_count/float(total_count);
            bad_ratio = bad_count/float(total_count);
            print('Good ratio: %s, Bad ratio: %s, Total count:%s' % (good_ratio, bad_ratio, total_count))

def test_comb(X_data, typelist, replacement = True):
    from sklearn.preprocessing import normalize
    if FUNCTION_TYPE == 'comb':
        from itertools import combinations, combinations_with_replacement
        if replacement == False:
            itr = combinations_with_replacement(typelist, n_feature)
            n_itr = len(typelist) * (len(typelist) + 1) * 0.5
        elif replacement == True:
            itr = combinations(typelist, n_feature)
            n_itr = len(typelist) * (len(typelist) - 1) * 0.5

        for i in range(int(n_itr)): # number of combination
            idx = itr.next()
            print(idx)

            direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH + '/result/' + PARAMETER_CONTROL + '/'  # + PARAMETER_CONTROL + '/'
            if not os.path.exists(direct):os.makedirs(direct)

            file_name = FUNCTION_TYPE + '-' + HOMOLOGY_TYPE + '-' + FILTRATION_TYPE + '-' + TEST + '.txt'
            file = open(direct + file_name, 'a')
            file.write('\n The combination is %s\n'%(idx,))
            file.close()

            X = X_data[idx[0]]['data']
            for i in range(1, len(idx)):
                X = np.concatenate((X, X_data[idx[i]]['data']), axis=1)
            # X = np.concatenate((X_data[idx[0]]['data'], X_data[idx[1]]['data']), axis=1)
            print('Idx is %s'%(idx,)),
            print('The shape of combined vector is %s'%(np.shape(X),))
            duplicate(X, 1)
            X = normalize(X, norm='l2', axis=NORMALIIZATION, copy=True)
            Y = Y_RANGE

            # final computation
            re = rfclf(X, Y)
            print(re)
            result_dict = clf_search_offprint(X, Y, 1)
            evaluate_best_estimator(result_dict, X, Y)
        print('Finishing testing feature combination. Exit.')

if __name__ =='__main__':
    from condition import theta
    tmp = initializeX(FILTRATION_TYPE, FUNCTION_TYPE, theta)
    X_data = tmp['X_data']; X = tmp['X']; typelist = tmp['typelist']
    print('Shape of X is %s' %(np.shape(X),))

    if replacement=='y':
        test_comb(X_data, typelist, replacement=True)
    elif replacement == 'n':
        test_comb(X_data, typelist, replacement=False)
    Y = Y_RANGE
    re = rfclf(X,Y)
    print(re)
    print(np.shape(X))
    result_dict = clf_search_offprint(X, Y, 1)
    evaluate_best_estimator(result_dict, X, Y)

def check():
        from sklearn import svm
        from sklearn.model_selection import cross_val_score
        clf = svm.SVC(kernel = 'linear', C = 100)
        for i in range(1,11): # different random states
            cross_val_score(clf, X, Y, cv = 10 , n_jobs=-1)
            clf.fit(X, Y)
        clf.fit()




