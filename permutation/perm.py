import sys
import argparse
import dionysus as d
import json
from joblib import Parallel, delayed
import numpy as np
import networkx as nx
from networkx.linalg.algebraicconnectivity import fiedler_vector

from aux.util import get_one_hyper, n_search
from aux.ricci import ricciCurvature
from aux.svm import evaluate_tda_kernel
from aux.persistence import set_hyperparameter, convert2nx, add_function_value
from aux.tools import load_data, dump_data, unit_vector, set_betalist, \
     diag2dgm, unzip_databundle, fv, flip_dgm, add_dgms, print_dgm, dgm2diag
from aux.helper import get_subgraphs, attribute_mean, get_diagram
from aux.util import fake_diagrams, kernel_parameter, make_hyper_direct, dgms_stats, graphassertion
from aux.sw import sw_parallel, dgms2swdgm
from aux.tools import load_graph
from aux.vector import clf_pdvector

def function_basis(g, allowed, norm_flag='no', recomputation_flag=False, transformation_flag=True):
    """

    :param g: nx graph
    :param allowed: filtration type, allowed = ['ricci', 'deg', 'hop', 'cc', 'fiedler']
    :param norm_flag: normalization flag
    :param recomputation_flag:
    :param transformation_flag: if apply linear/nonlinear transformation of filtration function
    :return: g with ricci, deg, hop, cc, fiedler computed
    """

    # to save recomputation. Look at the existing feature at first and then simply compute the new one.
    assert nx.is_connected(g)
    if len(g) < 3: return
    existing_features = [g.node[list(g.nodes())[0]].keys()]

    if not recomputation_flag:
        allowed = [feature for feature in allowed if feature not in existing_features]
    elif recomputation_flag:
        allowed = allowed

    def norm(g_, key, flag=norm_flag):
        if flag == 'no': return 1
        elif flag == 'yes': return np.max(np.abs(nx.get_node_attributes(g_, key).values())) + 1e-6
        else: raise('Error')

    # ricci
    g_ricci = g
    if 'ricciCurvature' in allowed:
        try:
            g_ricci = ricciCurvature(g, alpha=0.5, weight='weight')
            assert g_ricci.node.keys() == list(g.nodes())
            ricci_norm = norm(g, 'ricciCurvature', norm_flag)
            for n_ in g_ricci.nodes():
                g_ricci.node[n_]['ricciCurvature'] /= ricci_norm
        except:
            print('RicciCurvature Error for graph, set 0 for all nodes')
            for n in g_ricci.nodes():
                g_ricci.node[n]['ricciCurvature'] = 0

    # degree
    if 'deg' in allowed:
        deg_dict = dict(nx.degree(g_ricci))
        for n in g_ricci.nodes():
            g_ricci.node[n]['deg'] = deg_dict[n]
        deg_norm = norm(g_ricci, 'deg', norm_flag)
        for n in g_ricci.nodes():
            g_ricci.node[n]['deg'] /= np.float(deg_norm)

    # hop
    if 'hop' in allowed:
        distance = nx.floyd_warshall_numpy(g)  # return a matrix
        distance = np.array(distance)
        distance = distance.astype(int)
        if norm_flag == 'no': hop_norm = 1
        elif norm_flag == 'yes': hop_norm = np.max(distance)
        else: raise Exception('norm flag has to be yes or no')
        for n in g_ricci.nodes():
            # if g_ricci has non consecutive nodes, n_idx is the index of hop distance matrix
            n_idx = list(g_ricci.nodes).index(n)
            assert n_idx <= len(g_ricci)
            # print(n, n_idx)
            g_ricci.node[n]['hop'] = distance[n_idx][:] / float(hop_norm)

    # closeness_centrality
    if 'cc' in allowed:
        cc = nx.closeness_centrality(g)  # dict
        cc = {k: v / min(cc.values()) for k, v in cc.iteritems()}  # no normalization for debug use
        cc = {k: 1.0 / v for k, v in cc.iteritems()}
        for n in g_ricci.nodes():
            g_ricci.node[n]['cc'] = cc[n]

    # fiedler
    if 'fiedler' in allowed:
        fiedler = fiedler_vector(g, normalized=False)  # np.ndarray
        assert max(fiedler) > 0
        fiedler = fiedler / max(np.abs(fiedler))
        assert max(np.abs(fiedler)) == 1
        for n in g_ricci.nodes():
            n_idx = list(g_ricci.nodes).index(n)
            g_ricci.node[n]['fiedler'] = fiedler[n_idx]

    any_node = list(g_ricci.node)[0]
    if 'label' not in g_ricci.node[any_node].keys():
        for n in g_ricci.nodes():
            g_ricci.node[n]['label'] = 0  # add dummy
    else:  # contains label key
        assert 'label' in g_ricci.node[any_node].keys()
        for n in g_ricci.nodes():
            label_norm = 40
            if graph == 'dd_test': label_norm = 90
            g_ricci.node[n]['label'] /= float(label_norm)

    if 'deg' in allowed:
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='deg', cutoff=1, iteration=0)

        # better normalization, used to include 1_0_deg_std/ deleted now:
        if norm_flag == 'yes':
            for attr in ['1_0_deg_sum']:
                norm_ = norm(g_ricci, attr, norm_flag)
                for n in g_ricci.nodes(): g_ricci.node[n][attr] = g_ricci.node[n][attr] / float(norm_)

    if 'label' in allowed:
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='label', cutoff=1, iteration=0)
        for n in g_ricci.nodes():
            attribute_mean(g_ricci, n, key='label', cutoff=1, iteration=1)

    if 'cc_min' in allowed:
        for n in g_ricci.nodes(): attribute_mean(g_ricci, n, key='cc')

    if 'ricciCurvature_min' in allowed:
        for n in g_ricci.nodes(): attribute_mean(g_ricci, n, key='ricciCurvature')

    return g_ricci

def set_global_variable(graph_):
    '''
    :param graph_: graph data type
    :return:
    allowed = ['deg', 'ricciCurvature', 'label', 'cc']
    allowed_edge = ['edge_ricci_minmax', 'edge_ricci_ave', 'edge_jaccard_minmax', 'edge_jaccard_ave', 'edge_p_minmax', 'edge_p_ave']
    '''
    global allowed, allowed_edge, beta_dict, axis

    if graph_ == 'imdb_binary' or graph_ == 'imdb_multi':
        allowed = ['deg']
    else:
        allowed = ['deg']

    allowed_edge = ['edge_jaccard_minmax', 'edge_jaccard_ave']
    beta_dict = {0: 'deg', 1: 'ricciCurvature', 2: 'fiedler', 3: 'cc', 4: 'label'}
    axis = 1  # 0 normalize feature and 1 normalize each sample
    return allowed, allowed_edge, beta_dict, axis

def set_node_filtration_param(beta_, allowed_):
    print('Beta is %s' % beta_),
    if (str(type(beta_)) == "<type 'numpy.ndarray'>") and (1 in beta_):  # exclude hop related filtration
        beta_name = beta_dict[list(beta_).index(1)]
        if beta_name not in allowed_:
            pass
            # raise beta_name_not_in_allowed
        print('Filtration is %s' % beta_name)
        hop_flag = 'n'
        basep = 0
    else:
        hop_flag = 'y'
        assert type(beta_) == str
        assert beta_[0:3] == 'hop'
        beta_name = beta_
        basep = beta_[4:]
    return beta_name, hop_flag, basep

def compute_graphs_(allowed, graph_isomorphisim, graph, norm_flag='no', feature_addition_flag=False,
                    skip_dump_flag='yes'):
    (graphs_tmp, message) = load_data(graph, 'dgms_normflag_' + norm_flag, beta=-1, no_load='yes')
    if message == 'success':
        return graphs_tmp

    if graph_isomorphisim == 'off':
        (graphs_tmp, flag) = load_data(graph, 'graphs_', no_load='yes')
        if flag != 'success':
            from joblib import Parallel, delayed
            graphs_tmp = Parallel(n_jobs=-1, batch_size='auto')(
                delayed(pipeline0)(i, allowed, norm_flag=norm_flag, feature_addition_flag=feature_addition_flag) for i
                in range(len(data)))
            # assert 'cc' in graphs_tmp[0][0].node[0].keys()
            # assert 'deg' in graphs_tmp[0][0].node[0].keys()
            # assert 'cc' not in graphs_[0][0].node[0].keys()
            # assert 'deg' in graphs_[0][0].node[0].keys()
            dump_data(graph, graphs_tmp, 'graphs_', skip=skip_dump_flag)
        print()

    dump_data(graph, graphs_tmp, 'dgms_normflag_' + norm_flag, beta=-1, still_dump='yes', skip='yes')

    if (graph == 'imdb_binary') or (graph == 'imdb_multi') or (graph == 'dd_test') or (graph == 'protein_data') or (
            graph == 'collab'):
        uniform_norm_flag = True
    else:
        uniform_norm_flag = False
    if norm_flag == 'yes': uniform_norm_flag = False
    if uniform_norm_flag:
        anynode = list(graphs_tmp[0][0].nodes)[0]
        print(graphs_tmp[0][0].nodes[anynode])
        attribute_lists = graphs_tmp[0][0].nodes[anynode].keys()
        attribute_lists = [attribute for attribute in attribute_lists if attribute != 'hop']
        for attribute in attribute_lists:
            max_ = 0
            tmp_max_ = []
            min_ = 1
            tmp_min_ = []
            for i in range(len(graphs_tmp)):
                if len(graphs_tmp[i]) == 0:
                    print('skip graph %s' % i)
                    continue
                tmp_max_ += [np.max(nx.get_node_attributes(graphs_tmp[i][0], attribute).values())]  # catch exception
                # except:
                #     print (i, graphs_tmp[i])
                tmp_min_ += [np.min(nx.get_node_attributes(graphs_tmp[i][0], attribute).values())]
            from heapq import nlargest
            print nlargest(5, tmp_min_)[-1]
            denominator = max(nlargest(10, tmp_max_)[-1], nlargest(10, np.abs(tmp_min_))[-1]) + 1e-10
            # denominator = max(max(np.abs(tmp_max_)),max(np.abs(tmp_min_)))+1e-10
            print('Attribute and demoninator: ', attribute, denominator)

            for i in range(len(graphs_tmp)):
                if len(graphs_tmp[i]) == 0:
                    continue
                n_component = len(graphs_tmp[i])
                for comp in range(n_component):
                    for v in graphs_tmp[i][comp].nodes():
                        graphs_tmp[i][comp].nodes[v][attribute] = graphs_tmp[i][comp].nodes[v][attribute] / np.float(
                            denominator)
                        # graphs_tmp[i][comp].nodes[v][attribute + '_uniform'] = graphs_tmp[i][comp].nodes[v][attribute]/ np.float(denominator)
                        # print graphs_tmp[i][0].nodes[v][attribute],

    return graphs_tmp
    print('Finish pipeline 0(Convert to nx, calculate deg, ricci, hop...)')

def pipeline0(i, allowed, version=1, print_flag='False', norm_flag='no', feature_addition_flag=False):
    # basically two steps.
    # 1) convert data dict to netowrkx graph , get gi 2) calculate function on networkx graphs, get gi_s
    # version1: deal with chemical graphs
    # version2: deal with all non-isomorphic graphs # prepare data. Only execute once.

    assert 'data' in globals()
    assert version == 1
    if i % 50 == 0: print('*')
    if not feature_addition_flag: gi = convert2nx(data[i], i)

    if not feature_addition_flag:
        subgraphs = get_subgraphs(gi)
    else:
        assert 'graphs_' in globals().keys()
        subgraphs = [g.copy() for g in graphs_[i]]

    gi_subgraphs = [function_basis(subgraph, allowed, norm_flag=norm_flag) for subgraph in subgraphs]
    gi_subgraphs = [g for g in gi_subgraphs if g is not None]

    return gi_subgraphs

def pipeline1(i, beta=np.array([0, 0, 0, 0, 1]), hop_flag='n',
              basep=0, debug='off', rs=100, edge_fil='off'):
    '''

    :param i: i-th graph
    :param beta: [deg, ricci, fiedler, cc]
    :param hop_flag:
    :param basep:
    :param debug: flag
    :param rs: random seed
    :param edge_fil:
    :return:
    calculate persistence diagram of a graph (may disconneced)
    '''

    # data: mutag dict
    assert 'data' in globals().keys()
    assert 'graphs_' in globals().keys()

    subgraphs = []
    (dgm_, dgm_sub, dgm_super, epd_dgm) = (d.Diagram([(0, 0)]), d.Diagram([(0, 0)]), d.Diagram([(0, 0)]), d.Diagram([(0, 0)]))

    for k in range(len(graphs_[i])):
        # prepare
        if debug == 'on': print('Processing graph %s, subgraph %s' %(i, k))
        g = graphs_[i][k]
        graphassertion(g)

        g = fv(g, beta, hop_flag=hop_flag, basep=basep, rs=rs, edge_fil=edge_fil)  # belong to pipe1
        (g, fv_list) = add_function_value(g, fv_input='fv_test', edge_value='max')  # belong to pipe1
        dgm_sub = get_diagram(g, key='fv', subflag='True')

        (g, fv_list) = add_function_value(g, fv_input='fv_test', edge_value='min')  # belong to pipe1
        dgm_super = get_diagram(g, key='fv', subflag='False')
        dgm_super = flip_dgm(dgm_super)
        epd_dgm = get_diagram(g, key='fv', one_homology_flag=True)
        dgm = add_dgms(dgm_sub, dgm_super)
        dgm_ = add_dgms(dgm_, dgm)
        subgraphs.append(g)

    if i % 50 == 0: print('.'),
    if i % 100 == 0: print_dgm(dgm)

    return subgraphs, dgm_, dgm_sub, dgm_super, epd_dgm

def get_dgms(beta=np.array([0, 0, 0, 0, 1]), parallel='on', n_jobs=-1, hop_flag='n', basep=0, rs=100, edge_fil='off'):
    # batch version of pipeline1

    assert 'graphs_' in globals().keys()
    n = len(graphs_)

    if parallel == 'off':
        (dgms, graphs, sub_dgms, super_dgms, epd_dgms) = ([0] * n, [0] * n, [0] * n, [0] * n, [0] * n)
        for i in range(n):
            (graphs[i], dgms[i], sub_dgms[i], super_dgms[i], epd_dgms[i]) = \
            pipeline1(i, beta=beta, hop_flag=hop_flag,basep=basep, rs=rs, edge_fil=edge_fil)
        assert 0 not in dgms
        return [(graphs[i], dgms[i], sub_dgms[i], super_dgms[i], epd_dgms[i]) for i in range(n)]  # tuple of length 2

    elif parallel == 'on':
        if basep == 'a':
            assert hop_flag == 'y'
            x_ = Parallel(n_jobs=-1)(delayed(handle_i)(i, rs) for i in range(n))  # a list of tuples/ haven't handle this
            return x_
        else:
            return Parallel(n_jobs=n_jobs)(delayed(pipeline1)(i, beta=beta, hop_flag=hop_flag, basep=basep, rs=rs, edge_fil=edge_fil) for i in range(n))

    else:
        raise ('get_dgms parallel corner case')

def dgms_data(graph, beta, n_jobs, debug_flag, norm_flag='no', hop_flag='n', basep=-1, rs=100, edge_fil='off'):
    # for beta, get the corresponding dgm(sub/super dgms)
    # save it for future use if necessary

    if basep > 0: print('Using base point %s' % basep)
    assert 'graphs_' in globals().keys()
    (graphs, flag1) = load_data(graph, 'graphs', beta, no_load='yes')
    (dgms_, flag2) = load_data(graph, 'dgms_', beta, no_load='yes')

    if flag2 == 'success':
        dgms = Parallel(n_jobs=-1)(delayed(diag2dgm)(dgms_[i]) for i in range(len(dgms_)))

    elif (flag1 != 'success') or (flag2 != 'success'):
        print('Computing graphs and dgms...')
        # haven't handle parallel case
        databundle = get_dgms(beta=beta, hop_flag=hop_flag, basep=basep, rs=rs, edge_fil=edge_fil)
        (graphs, dgms, sub_dgms, super_dgms, epd_dgms) = unzip_databundle(databundle)
        dump_data(graph, graphs, 'graphs', beta, skip='yes') # dump data

        for tmp in ['dgms', 'sub_dgms', 'super_dgms']:
            save_data = eval(tmp)
            data_ = Parallel(n_jobs=-1)(delayed(dgm2diag)(save_data[i]) for i in range(len(save_data)))
            dump_data(graph, data_, tmp, beta)

    else:
        raise Exception('Corner case')

    return graphs, dgms, sub_dgms, super_dgms, epd_dgms

def select_dgms(dgms, sub_dgms, super_dgms, epd_dgms, homtype='0'):
    if homtype == '0':
        return dgms
    elif homtype == '0-':
        return sub_dgms
    elif homtype == '0+':
        return super_dgms
    elif homtype == '1':
        return epd_dgms
    elif homtype == '01':
        assert len(epd_dgms) == len(dgms)
        sum_dgms = [add_dgms(dgms[i], epd_dgms[i]) for i in range(len(dgms))]
        return sum_dgms

parser = argparse.ArgumentParser()
parser.add_argument('--graph', default='mutag', help="mutag, ptc, reddit...")
parser.add_argument('--dgmtype', default='normal', help="normal or fake")
parser.add_argument('--kerneltype', default='sw', help="sw, pss, wg")
parser.add_argument('--homtype', default='0', help="0, 1, 01, 0-, 0+")

parser.add_argument('--comp', dest='computation_flag', action='store_true')
parser.add_argument('--no-comp', dest='computation_flag', action='store_false')
parser.set_defaults(computation_flag=True)

parser.add_argument('--eval', dest='eval_flag', action='store_true')
parser.add_argument('--no-eval', dest='eval_flag', action='store_false')
parser.set_defaults(eval_flag=True)

parser.add_argument('--load', dest='load_flag', action='store_true')
parser.add_argument('--no-load', dest='load_flag', action='store_false')
parser.set_defaults(load_flag=False)

if __name__ == '__main__':
    args = parser.parse_args()
    graph = args.graph
    homtype = args.homtype
    suffix = args.dgmtype + '_' + homtype + '_'
    computation_flag = args.computation_flag
    evaluation_flag = args.eval_flag
    load_flag = args.load_flag

    print('graph is %s' % graph)
    hyperparameter = set_hyperparameter()
    (allowed, allowed_edge, beta_dict, axis) = set_global_variable(graph)
    (hyperparameter_flag, norm_flag, loss_type, n_runs, pd_flag, multi_cv_flag, n_jobs, debug_flag, graph_isomorphisim,
     edge_fil, dynamic_range_flag) = hyperparameter
    (data, Y) = load_graph(graph)
    n = len(Y)

    if not load_flag:
        graphs_backup = compute_graphs_(['deg', 'ricciCurvature', 'cc'], graph_isomorphisim, graph, norm_flag='yes',
                                        feature_addition_flag=False)
        graphs_ = graphs_backup
        print(graphs_backup[0][0].node[0])

    beta = unit_vector(5, 3)
    norm_flag = 'yes'
    betalist = set_betalist(allowed)
    for beta in betalist:
        assert len(betalist) == 3
        (beta_name, hop_flag, basep) = set_node_filtration_param(beta, allowed)

        if not load_flag:
            (graphs, dgms, sub_dgms, super_dgms, epd_dgms) = dgms_data(graph, beta, n_jobs, debug_flag,
                                                                       hop_flag=hop_flag, basep=basep, edge_fil='off')
            dgms = select_dgms(dgms, sub_dgms, super_dgms, epd_dgms, homtype=homtype)
            if args.dgmtype == 'fake':
                dgms = fake_diagrams(dgms, true_dgms=dgms, seed=45)
            dgmstat = dgms_stats(dgms)
            best_vec_result = 0; epd_flag = False; ax = 1; rf_flag = 'y'
            pd_vector_data = clf_pdvector(best_vec_result, (sub_dgms, super_dgms, dgms, epd_dgms), beta, Y,
                                          epd_flag=epd_flag, pd_flag='True', print_flag='off', nonlinear_flag=True,
                                          axis=ax, rf_flag=rf_flag,
                                          dynamic_range_flag=dynamic_range_flag)  # use pd vector as baseline
            print pd_vector_data
        sys.exit()
        for kernel_type in [args.kerneltype]:
            searchrange = kernel_parameter(kernel_type)
            args_ = {}
            for i in range(n_search(searchrange)):
                args_ = get_one_hyper(searchrange, method=kernel_type, idx=i, tf=suffix, filtration=beta_name)

                # load
                if load_flag:
                    direct = make_hyper_direct(beta_name=beta_name, dataset=graph, make_flag=False, **args_)
                    with open(direct + suffix + 'eval.json', 'r') as f:
                        jdict = json.load(f)
                    print jdict['eval']

                # computation
                if computation_flag:
                    (tda_kernel, t1) = sw_parallel(dgms2swdgm(dgms), dgms2swdgm(dgms), parallel_flag=True,
                                                   kernel_type=kernel_type, **args_)

                    # save
                    direct = make_hyper_direct(beta_name=beta_name, dataset=graph, make_flag=True, **args_)
                    np.save(direct + suffix + 'kernel', tda_kernel)
                    with open(direct + suffix + 'param.json', 'wa') as fp:
                        json.dump(args_, fp, indent=1)

                # evaluation
                if evaluation_flag:
                    direct = make_hyper_direct(beta_name=beta_name, dataset=graph, make_flag=False, **args_)
                    try:
                        tda_kernel = np.load(direct + suffix + 'kernel.npy')
                    except:
                        raise Exception('No precomputed kernel at ', direct + suffix + 'kernel.npy')

                    best_result_so_far = (0, 0, {}, 0)
                    (precision, std, kparam, time) = evaluate_tda_kernel(tda_kernel, Y, best_result_so_far)
                    eval_result = {'precision': precision, 'std': std, 'kernel_hyperparameter': kparam,
                                   'evaluation_time': time, 'tf': suffix}
                    with open(direct + suffix + 'eval.json', 'wa') as fp:
                        json.dump({'eval': eval_result, 'dgmstat': dgmstat}, fp, indent=1)

