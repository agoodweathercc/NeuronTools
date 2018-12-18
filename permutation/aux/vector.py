import numpy as np
import dionysus as d
import time
import sys
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
import sklearn_tda as tda

from .sw import dgms2swdgm
from .tools import dgms_summary, add_dgms, precision_format, evaluate_best_estimator, remove_zero_col, normalize_
from .svm import clf_search_offprint, rfclf

def dgm2diag(dgm):
    assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    return diag

def filterdict(dict, keys):
    # keys: a list of keys
    dict_ = {}
    for key, val in dict.items():
        if key in keys: dict_[key] = val
    return dict_

def dgm_vec(diags, vec_type='pi', axis=1, **kwargs):
    print kwargs
    t1 = time.time()
    def arctan(C, p):
        return lambda x: C * np.arctan(np.power(x[1], p))

    if vec_type == 'pi':
        diagsT = tda.DiagramPreprocessor(use=True, scaler=tda.BirthPersistenceTransform()).fit_transform(diags)
        kwargs = filterdict(kwargs, ['bandwidth', 'weight', 'im_range', 'resolution'])
        kwargs['weight'] = arctan(kwargs['weight'], kwargs['weight']+1)
        print kwargs
        PI = tda.PersistenceImage(**kwargs)
        # PI = tda.PersistenceImage(bandwidth=1.0, weight=arctan(1.0, 1.0), im_range=[0, 1, 0, 1], resolution=[25, 25])
        res = PI.fit_transform(diagsT)

    elif vec_type == 'pl':
        LS = tda.Landscape(num_landscapes=5, resolution=100)
        res = LS.fit_transform(diags)

    t2 = time.time()
    t = precision_format((t2 - t1), 1)
    return remove_zero_col(normalize_(res, axis=axis)), t

def unwrap_pdvector(*arg, **kwarg):
    return pdvector.persistence_vector(*arg, **kwarg)

class pdvector():
    def __init__(self, dgm, dynamic_range_flag = True):
        self.dgm = dgm
        self.dynamic_range_flag = dynamic_range_flag

    def data_interface(self, dgm, dynamic_range_flag=True):
        # from dgm to data/max/min
        for p in dgm: assert p.death >= p.birth
        data = [tuple(i) for i in dgm2diag(dgm)]
        try:
            [list1, list2] = zip(*data);
        except:
            print('Problem')
            list1 = [0];
            list2 = [1e-5]  # adds a dummy 0

        if dynamic_range_flag == True:
            min_ = min(min(list1), min(list2))
            max_ = max(max(list1), max(list2))
            std_ = (np.std(list1) + np.std(list2)) / 2.0
        elif dynamic_range_flag == False:
            min_ = -5
            max_ = 5
            std_ = 3

        return {'data': data, 'max': max_ + std_, 'min': min_ - std_}

    @staticmethod
    def rotate_data(data, super_check):
        """
        :param data:
        :return: a list of tuples
        """

        def rotate(x, y):
            return np.sqrt(2) / 2 * np.array([x + y, -x + y])

        def flip(x, y):
            assert x >= y
            return np.array([y, x])

        length = len(data)
        rotated = []

        for i in range(0, length, 1):
            if super_check == True: data[i] = flip(data[i][0], data[i][1])
            point = rotate(data[i][0], data[i][1]);
            point = (point[0], point[1])
            rotated.append(point)
        return rotated
    @staticmethod
    def draw_data(data, imax, imin, discrete_num=500):
        """
        :param data: a list of tuples
        :return: a dictionary: vector of length 1000

        """
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2) + 0.000001))

        discrete_num = discrete_num
        assert imax >= imin
        distr = np.array([0] * discrete_num)
        par = data
        for x, y in par:
            mu = x;
            sigma = y / 3.0
            distr = distr + y * gaussian(np.linspace(imin - 1, imax + 1, discrete_num), mu, sigma)
        return distr

    def persistence_vector(self, dgm, discete_num=500, debug_flag=False):
        ## here filtration only takes sub or super
        result = self.data_interface(dgm, dynamic_range_flag=self.dynamic_range_flag)
        data = result['data']
        imax = result['max']
        imin = result['min']
        if debug_flag: print(imax, imin)
        data = self.rotate_data(data, super_check=False)
        vector = self.draw_data(data, imax, imin, discrete_num=discete_num)
        vector = np.array(vector).reshape(1, len(vector))
        return vector

    def persistence_vectors(self, dgms, debug='off', axis=1, dynamic_range_flag=True):
        start = time.time()
        n1 = len(dgms)
        n2 = np.shape(self.persistence_vector(dgms[0]))[1]
        X = np.zeros((n1, n2))
        X_list = Parallel(n_jobs=-1)(delayed(unwrap_pdvector)(self, dgms[i]) for i in range(len(dgms)))
        for i in range(n1): X[i] = X_list[i]
        if debug == 'on': print('persistence_vectores takes %s' % (time.time() - start))
        X = normalize(X, norm='l2', axis=axis, copy=True)
        return X


def clf_pdvector(best_vec_result, (sub_dgms, super_dgms, dgms, epd_dgms),
                 beta, Y, epd_flag=False, pvec_flag=False, vec_type='pi',
                 pd_flag='False', multi_cv_flag=False, print_flag='off', nonlinear_flag=True,
                 axis=1, rf_flag='y', dynamic_range_flag=True, **kwargs):

    # pd vector classification
    assert pd_flag == 'True'
    if epd_flag == False:
        (stat1, stat2) = dgms_summary(dgms)  # with and without multiplicity
    else:
        tmp_dgms = [add_dgms(dgms[i], epd_dgms[i]) for i in range(len(dgms))]
        (stat1, stat2) = dgms_summary(tmp_dgms)  # with and without multiplicity

    if pvec_flag == False:
        vct_ = time.time()
        x_sub = persistence_vectors(sub_dgms, axis=axis, dynamic_range_flag=dynamic_range_flag)
        x_super = persistence_vectors(super_dgms, axis=1, dynamic_range_flag=dynamic_range_flag)
        if epd_flag == False:
            X = np.concatenate((x_sub, x_super), axis=1)
        else:
            x_epd = persistence_vectors(epd_dgms, axis=1, dynamic_range_flag=dynamic_range_flag)
            X = np.concatenate((x_sub, x_super, x_epd), axis=1)
        vct = precision_format(time.time() - vct_, 1)

    elif pvec_flag == True:  # needs to change here
        (X_sub, vct) = dgm_vec(dgms2swdgm(sub_dgms), vec_type=vec_type, axis=axis)
        (X_super, vct) = dgm_vec(dgms2swdgm(super_dgms), vec_type=vec_type, axis=axis)
        if epd_flag == False:
            X = np.concatenate((X_sub, X_super), axis=1)
        elif epd_flag == True:
            (X_epd, vct) = dgm_vec(dgms2swdgm(epd_dgms), vec_type=vec_type, axis=axis)
            X = np.concatenate((X_sub, X_super, X_epd), axis=1)
    print('Shape of X as PD vector is', (np.shape(X)))

    if rf_flag == 'y':
        rf_ = rfclf(X, Y, multi_cv_flag=multi_cv_flag)
        print rf_
        rf_ = ["{0:.1f}".format(100 * i) for i in rf_]
        return (rf_, (0, 0))

    rf_ = rfclf(X, Y, multi_cv_flag=multi_cv_flag)  # print(rf_)
    t1 = time.time()
    grid_search_re = clf_search_offprint(X, Y, random_state=2, nonlinear_flag=nonlinear_flag, print_flag=print_flag)
    if grid_search_re['score'] < best_vec_result - 2:
        print ('Saved one unnecessary evaluation of bad kernel ')
        return
    cv_score = evaluate_best_estimator(grid_search_re, X, Y, print_flag=print_flag)
    t2 = time.time()
    print('Finish calculating persistence diagrams\n')
    rf_ = ["{0:.1f}".format(100 * i) for i in rf_]
    cv_score = ["{0:.1f}".format(100 * i) for i in cv_score]
    return (rf_, cv_score, str(round(t2 - t1)), str(grid_search_re['param']),
            stat1, stat2, str(vct) + ' epd_flag = ' + str(epd_flag))

if __name__ == '__main__':
    dgm = d.Diagram([(2,3), (3,4)])
    pdv = pdvector(dgm)
    feature = pdv.persistence_vector(dgm)
    features = pdv.persistence_vectors([dgm]*10000)
    print np.shape(features)
    assert (features[300] == features).all()
    sys.exit()



    diag = dgms2swdgm([dgm])
    print diag
    dgm_vec(diag, vec_type='pi', axis=1, **{'weight': 1})
    print(feature)