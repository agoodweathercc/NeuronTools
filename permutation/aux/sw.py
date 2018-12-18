import numpy as np

def dgm2diag(dgm):
    assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
    diag = list()
    for pt in dgm:
        # print(pt),
        # print(type(pt))
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    return diag

def precision_format(nbr, precision=1):
    # assert type(nbr)==float
    return  round(nbr * (10**precision))/(10**precision)

def assert_sw_dgm(dgms):
    # check sw_dgm is a list array
    # assert_sw_dgm(generate_swdgm(10))
    assert type(dgms)==list
    for dgm in dgms:
        assert np.shape(dgm)[1]==2

def sw(dgms1, dgms2, parallel_flag=False, kernel_type='sw', n_directions=10, bandwidth=1.0, K=1, p = 1):
    def arctan(C, p):
        return lambda x: C * np.arctan(np.power(x[1], p))

    import sklearn_tda as tda
    if parallel_flag==False:
        if kernel_type=='sw':
            tda_kernel = tda.SlicedWassersteinKernel(num_directions=n_directions, bandwidth=bandwidth)
        elif kernel_type=='pss':
            tda_kernel = tda.PersistenceScaleSpaceKernel(bandwidth=bandwidth)
        elif kernel_type == 'wg':
            tda_kernel = tda.PersistenceWeightedGaussianKernel(bandwidth=bandwidth, weight=arctan(K, p))
        else:
            print ('Unknown kernel')

        diags = dgms1; diags2 = dgms2
        X = tda_kernel.fit(diags)
        Y = tda_kernel.transform(diags2)
        return Y

def sw_parallel(dgms1, dgms2, parallel_flag=True, kernel_type='sw', n_directions=10, bandwidth=1.0, K = 1, p = 1, granularity=25, **kwargs):
    import time
    t1 = time.time()
    assert_sw_dgm(dgms1)
    assert_sw_dgm(dgms2)
    from joblib import Parallel, delayed
    n1 = len(dgms1); n2 = len(dgms2)
    kernel = np.zeros((n2, n1))

    if parallel_flag==False:         # used as verification
        for i in range(n2):
            kernel[i] = sw(dgms1, [dgms2[i]], kernel_type=kernel_type, n_directions=n_directions, bandwidth=bandwidth)
    if parallel_flag==True:
        # parallel version
        kernel = Parallel(n_jobs=-1)(delayed(sw)(dgms1, dgms2[i:min(i+granularity, n2)], kernel_type=kernel_type, n_directions=n_directions, bandwidth=bandwidth, K=K, p=p) for i in range(0, n2, granularity))
        kernel = (np.vstack(kernel))
    return (kernel/float(np.max(kernel)), precision_format(time.time()-t1, 1))

def dgms2swdgm(dgms):
    swdgms=[]
    for dgm in dgms:
        diag = dgm2diag(dgm)
        swdgms += [np.array(diag)]
    return swdgms
