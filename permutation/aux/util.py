import dionysus as d
import networkx as nx
import numpy as np
from numpy import random
import sys
import json

from aux.persistence import set_hyperparameter, threshold_data, load_graph, convert2nx, check_global_safety, add_function_value
from aux.tools import load_data, dump_data, read_all_graphs, generate_graphs, unit_vector, set_betalist, \
                        beta_name_not_in_allowed,  diag2dgm, unzip_databundle, fv, flip_dgm, add_dgms, print_dgm, dgm2diag
from helper import get_subgraphs, attribute_mean, get_diagram
from aux.tools import  make_direct


def fake_diagram(cardinality = 2, seed=42, true_dgm = 'null'):
    random.seed(seed)
    if true_dgm!='null':
        array_tmp = dgm2diag(true_dgm)
        sample_pool = [p[0] for p in array_tmp] + [p[1] for p in array_tmp]
    else:
        raise Exception('true_dgm must not be null')
    try:
        sample = np.random.choice(sample_pool, size=2*cardinality, replace=False)
    except:
        sample = np.random.choice(sample_pool, size=2 * cardinality, replace=True)
    assert set(sample).issubset(set(sample_pool))
    dgm = []
    for i in range(0, len(sample),2):
        x_ = sample[i]
        y_ = sample[i+1]
        dgm.append((min(x_, y_), max(x_, y_)+1e-3))
    return d.Diagram(dgm)


def fake_diagrams(dgms, true_dgms = ['null']*10000, seed=45):
    fake_dgms = []
    for i in range(len(dgms)):
        cardin = len(dgms[i])
        if len(dgms[i])==0:
             fake_dgms.append(d.Diagram([(0,0)]))
             continue
        print cardin
        tmp_dgm = fake_diagram(cardinality = cardin, seed=seed, true_dgm=true_dgms[i])
        fake_dgms.append(tmp_dgm)
    return fake_dgms

# fake_diagram(cardinality = 100, attribute='deg', seed=42, true_dgm = dgms[100])
# fake_diagrams(graphs_, dgms, true_dgms = dgms, attribute='deg', seed=45)

def kernel_parameter( kernel_type='sw'):
    if kernel_type=='sw':
        # bw = [0.01, 0.1, 1, 10, 100]
        bw = [1, 0.1, 10, 0.01, 100, 1000, 0.001]
        k = [1]; p=[1];
    elif kernel_type == 'pss':
        # bw = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100, 500, 1000]
        bw = [1, 5e-1, 5, 1e-1, 10, 5e-2, 50, 1e-2, 100, 5e-3, 500, 1e-3, 1000]
        k = [1]; p = [1];
    elif kernel_type == 'wg':
        bw = [1, 10, 0.1, 100, 0.01]
        # k = [1];
        # p = [1];
        k = [ 1, 10, 0.1];
        p = [ 1, 10, 0.1];
    return {'bw':bw, 'K':k, 'p':p}

def make_hyper_direct(method ='sw', beta_name = 'deg', dataset ='mutag', make_flag=False, **kwargs):
    # generate the directory for a particular set of hyper-parameters

    parameters = kwargs # a dictionary
    if method == 'sw':
        hyper_parameter_orderlist = ['bw'] # can be extended
    elif method == 'pss':
        hyper_parameter_orderlist = ['bw']
    elif method == 'wg':
        hyper_parameter_orderlist = ['bw', 'K', 'p']
    elif method == 'pdvector':
        hyper_parameter_orderlist = ['d', 'beta']  # can be extended
    else:
        raise Exception('Unknown method!')

    # make directory for some hyperparameter combination
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/experiments/' + dataset +  '/' + method + '/' + beta_name + '/'
    for parameter in hyper_parameter_orderlist:
        if parameter in parameters.keys():
            direct += parameter + '_' + str(parameters[parameter]) + '/'

    if make_flag == True:
        make_direct(direct)

    return direct

def my_product(inp):
    from itertools import product
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

def n_search(searchrange):
    n = np.product(map(len, searchrange.values()))
    return n

def get_one_hyper(searchrange, method = 'sw', filtration = 'deg', tf='normal', idx=1): # need to modify hyper-param
    # from searchrange, get 1 hyperparamater combination
    # searchrange example: {'K': [1], 'bw': [1, 0.1, 10, 0.01, 100, 0.01], 'p': [1]}
    n = np.product(map(len, searchrange.values()))
    assert (n-1) >= idx
    result =  list(my_product(searchrange))[idx];
    result['n_directions'] = 10
    result['method'] = method
    result['filtraiton'] = filtration
    result['T/F'] = tf
    return result

def dgms_stats(dgms):
    lenlist = map(len, dgms)
    return {'min': np.min(lenlist), 'max': np.max(lenlist),
            'ave': np.average(lenlist), 'std': np.std(lenlist)}

def load_kernel( graph='mutag', method='sw', normal_flag = True, debug_flag = False, filtration = 'cc', **kwargs):
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/experiments/' # \
             # 'mutag/sw/cc/bw_0.1/normal_kernel.npy'
    import os
    direct += os.path.join(graph, method, filtration, '')
    for key, val in kwargs.items():
        direct += os.path.join(key + '_' + str(val), '')
    if debug_flag:
        print (direct)
    if normal_flag == True:
        file = 'normal_kernel.npy'
    else:
        file = 'fake_kernel.npy'
    kernel = np.load(direct + file)
    assert np.max(np.abs((kernel - kernel.T))) < 1e-5
    return kernel
# args_ = {'bw': 0.1}
# load_kernel( graph='mutag', method='sw', normal_flag = True, filtration = 'cc', **args_)

# for idx in range(6):
#     print get_one_hyper(sw_param, idx = idx)
def load_stat(direct, graph, ):
    direct = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation'

    with open('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/permutation/experiments/ptc/sw/deg/bw_10/fake_0_eval.json','r') as f:
        jdict = json.load(f)

def graphassertion(g):
    assert str(type(g)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
