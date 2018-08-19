import sys
sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/pythoncode')
from cycle_tools import print_dgm, dgm2diag, diag2dgm

def test_viz():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    import numpy

    data = {(0.0, 0.0, 1.0): 0.5874125874125874,
            (0.0, 0.2, 0.8): 0.6593406593406593,
            (0.0, 0.25, 0.75): 0.6783216783216783,
            (0.0, 0.3333, 0.6667): 0.6933066933066933,
            (0.0, 0.4, 0.6): 0.7002997002997003,
            (0.0, 0.4286, 0.5714): 0.7072927072927073,
            (0.0, 0.5, 0.5): 0.7062937062937062,
            (0.0, 0.5714, 0.4286): 0.7042957042957043,
            (0.0, 0.6, 0.4): 0.7022977022977023,
            (0.0, 0.6667, 0.3333): 0.7102897102897103,
            (0.0, 0.75, 0.25): 0.7092907092907093,
            (0.0, 0.8, 0.2): 0.7052947052947053,
            (0.1111, 0.4444, 0.4444): 0.6933066933066933,
            (0.125, 0.375, 0.5): 0.6893106893106893,
            (0.125, 0.5, 0.375): 0.6993006993006993,
            (0.1429, 0.2857, 0.5714): 0.6883116883116883,
            (0.1429, 0.4286, 0.4286): 0.6923076923076923,
            (0.1429, 0.5714, 0.2857): 0.7032967032967034,
            (0.1667, 0.1667, 0.6667): 0.6783216783216783,
            (0.1667, 0.3333, 0.5): 0.6813186813186813,
            (0.1667, 0.5, 0.3333): 0.6993006993006993,
            (0.1667, 0.6667, 0.1667): 0.6823176823176823,
            (0.2, 0.0, 0.8): 0.6833166833166833,
            (0.2, 0.2, 0.6): 0.6783216783216783,
            (0.2, 0.4, 0.4): 0.7012987012987013,
            (0.2, 0.6, 0.2): 0.6833166833166833,
            (0.2, 0.8, 0.0): 0.6693306693306693,
            (0.2222, 0.3333, 0.4444): 0.6923076923076923,
            (0.2222, 0.4444, 0.3333): 0.6973026973026973,
            (0.25, 0.0, 0.75): 0.6853146853146853,
            (0.25, 0.25, 0.5): 0.6863136863136863,
            (0.25, 0.375, 0.375): 0.7002997002997003,
            (0.25, 0.5, 0.25): 0.6973026973026973,
            (0.25, 0.75, 0.0): 0.6663336663336663,
            (0.2727, 0.3636, 0.3636): 0.7012987012987013,
            (0.2857, 0.1429, 0.5714): 0.6793206793206793,
            (0.2857, 0.2857, 0.4286): 0.6943056943056943,
            (0.2857, 0.4286, 0.2857): 0.6943056943056943,
            (0.2857, 0.5714, 0.1429): 0.6763236763236763,
            (0.3, 0.3, 0.4): 0.6993006993006993,
            (0.3, 0.4, 0.3): 0.6993006993006993,
            (0.3333, 0.0, 0.6667): 0.6753246753246753,
            (0.3333, 0.1667, 0.5): 0.6833166833166833,
            (0.3333, 0.2222, 0.4444): 0.6913086913086913,
            (0.3333, 0.3333, 0.3333): 0.7012987012987013,
            (0.3333, 0.4444, 0.2222): 0.6853146853146853,
            (0.3333, 0.5, 0.1667): 0.6833166833166833,
            (0.3333, 0.6667, 0.0): 0.6703296703296703,
            (0.3636, 0.2727, 0.3636): 0.7012987012987013,
            (0.3636, 0.3636, 0.2727): 0.6903096903096904,
            (0.375, 0.125, 0.5): 0.6843156843156843,
            (0.375, 0.25, 0.375): 0.7022977022977023,
            (0.375, 0.375, 0.25): 0.6893106893106893,
            (0.375, 0.5, 0.125): 0.6803196803196803,
            (0.4, 0.0, 0.6): 0.6773226773226774,
            (0.4, 0.2, 0.4): 0.6983016983016983,
            (0.4, 0.3, 0.3): 0.6933066933066933,
            (0.4, 0.4, 0.2): 0.6873126873126874,
            (0.4, 0.6, 0.0): 0.6913086913086913,
            (0.4286, 0.0, 0.5714): 0.6803196803196803,
            (0.4286, 0.1429, 0.4286): 0.6893106893106893,
            (0.4286, 0.2857, 0.2857): 0.6993006993006993,
            (0.4286, 0.4286, 0.1429): 0.6823176823176823,
            (0.4286, 0.5714, 0.0): 0.6933066933066933,
            (0.4444, 0.1111, 0.4444): 0.6903096903096904,
            (0.4444, 0.2222, 0.3333): 0.6983016983016983,
            (0.4444, 0.3333, 0.2222): 0.6873126873126874,
            (0.4444, 0.4444, 0.1111): 0.6853146853146853,
            (0.5, 0.0, 0.5): 0.6893106893106893,
            (0.5, 0.125, 0.375): 0.7022977022977023,
            (0.5, 0.1667, 0.3333): 0.7012987012987013,
            (0.5, 0.25, 0.25): 0.6933066933066933,
            (0.5, 0.3333, 0.1667): 0.6863136863136863,
            (0.5, 0.375, 0.125): 0.6883116883116883,
            (0.5, 0.5, 0.0): 0.7012987012987013,
            (0.5714, 0.0, 0.4286): 0.6913086913086913,
            (0.5714, 0.1429, 0.2857): 0.6943056943056943,
            (0.5714, 0.2857, 0.1429): 0.6873126873126874,
            (0.5714, 0.4286, 0.0): 0.6933066933066933,
            (0.6, 0.0, 0.4): 0.6983016983016983,
            (0.6, 0.2, 0.2): 0.6853146853146853,
            (0.6, 0.4, 0.0): 0.7022977022977023,
            (0.6667, 0.0, 0.3333): 0.7042957042957043,
            (0.6667, 0.1667, 0.1667): 0.6863136863136863,
            (0.6667, 0.3333, 0.0): 0.7042957042957043,
            (0.75, 0.0, 0.25): 0.7042957042957043,
            (0.75, 0.25, 0.0): 0.7092907092907093,
            (0.8, 0.0, 0.2): 0.7102897102897103,
            (0.8, 0.2, 0.0): 0.7122877122877123,
            (1.0, 0.0, 0.0): 0.7102897102897103}

    DATA = numpy.random.rand(20, 3)
    Xs = np.array([i[0] for i in data.keys()])  # DATA[:,0]
    Ys = np.array([i[1] for i in data.keys()])  # DATA[:,0]
    ZZ = np.array([i[2] for i in data.keys()])  # DATA[:,0]
    Zs = np.array(data.values())
    for i in range(len(Xs)):
        assert data[(Xs[i], Ys[i], ZZ[i])] == Zs[i]

    # ======
    ## plot:

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    # 2
    ax = fig.add_subplot(222, projection='3d')
    ax.plot_trisurf(Xs, Ys, Zs, linewidth=0.2, antialiased=True)
    # 3
    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(Xs, Ys, Zs)
    # Axes3D.scatter

    fig.tight_layout()

    plt.show()  # or:
    # fig.savefig('3D.png')
# test_viz()
def dgm_distinct(dgm):
    diag = dgm2diag(dgm)
    diag = dgm2diag(dgm)
    distinct_list = [i[0] for i in diag]
    distinct_list += [i[1] for i in diag]
    distinct_list.sort()
    return distinct_list
def read_neuronPD(id):
    """
    Read data from NeuronTools
    :return: a list of tuples
    """
    from subprocess import check_output
    try:
        file_directory = check_output('find /Users/admin/Documents/osu/Research/DeepGraphKernels/linux_sync/deep-persistence/NeuronTools/persistence_diagrams/3_data_1268 -depth 1 -name ' +  str(id) + '_*', shell=True).split('\n')[:-1][0]
    except:
        file_directory = check_output('find /home/cai.507/Documents/DeepLearning/deep-persistence/NeuronTools/persistence_diagrams/3_data_1268 -maxdepth 1 -name ' + str('\'') + str(id) + '_*' + str('\''), shell=True).split('\n')[:-1][0]
    print('The existing PD name is %s '%file_directory[-30:])
    file = open(file_directory,"r")
    data = file.read(); data = data.split('\r\n');
    Data = []
    # this may need to change for different python version
    for i in range(0,len(data)-1):
        data_i = data[i].split(' ')
        data_i_tuple = (float(data_i[0]), float(data_i[1]))
        Data = Data + [data_i_tuple]
    return Data
def get_swc_files():
    filename1 = '/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/NeuronTools/Experiments/data/data_1268/test.out'
    filename2 = '/home/cai.507/Documents/DeepLearning/deep-persistence/NeuronTools/Experiments/data/data_1268/test.out'
    try:
        file = open(filename1, 'r')
    except:
        file = open(filename2, 'r')
    data = file.read(); data = data.split('\n'); data = data[:-1]
    assert len(data) == 1268
    return data
    # pass
    # from subprocess import call, check_output
    # try:
    #     files = check_output('find /Users/admin/Documents/osu/Research/DeepGraphKernels/linux_sync/deep-persistence/NeuronTools/Experiments/data/data_1268 -name *.swc', shell=True).split('\n')[:-1]
    # except:
    #     files = check_output('find /home/cai.507/Documents/DeepLearning/deep-persistence/NeuronTools/Experiments/data/data_1268 -name \'*.swc\'', shell=True).split('\n')[:-1]
    # assert len(files) == 1268
    # return files
def get_swc_file(i):
    files = get_swc_files()
    print ('swc file is %s' %(files[i-1][-30:]))
    return files[i-1]
def test_correspondence():
    for id in [1, 709, 711, 730, 780, 1000, 1100, 1200]:
        get_swc_file(id)
        null = read_neuronPD(id)
        print ('\n')
def read_data(files, i):
        """
        Read data from NeuronTools
        """
        # file_directory = '/Users/admin/Documents/osu/Research/DeepGraphKernels/linux_sync/deep-persistence/NeuronTools_origin/Experiments/data/data_1268/neuron_nmo_principal/hay+markram/CNG version/cell4.CNG.swc'
        file_directory = files[i-1]
        # file_directory = '/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/NeuronTools/Test/2000_manual_tree_T1.swc'
        print ('The %s -th raw data is %s'%(i, file_directory))
        file = open(file_directory, "r")
        data = file.read();
        data = data.split('\r\n');
        # Data = []
        # this may need to change for different python version
        # for i in range(0, len(data) - 1):
        #     data_i = data[i].split(' ')
        #     data_i_tuple = (float(data_i[0]), float(data_i[1]))
        #     Data = Data + [data_i_tuple]
        return data
def get_df(files, i):
    # i = 924
    data = read_data(files, i)
    data = data[0]
    data = data.split('\n')
    for j in range(30):
        if data[j].startswith('#'):
            idx = j
    data = data[idx+1:]
    if data[0].startswith(' '):
        for i in range(len(data)):
            data[i] = data[i][1:] # remove the first letter in the string
    assert data[0].startswith('1')

    length = len(data)
    import numpy as np
    data_array = np.array([-1] * 7).reshape(1, 7)
    for i in data[0:length - 1]:
        i = i.split(' ')
        if i[-1] == '':
            i = i.remove('')
        ary = np.array([float(s) for s in i]).reshape(1, 7)
        data_array = np.concatenate((data_array, ary), axis=0)
    import pandas as pd
    colnames = ['id', 'structure', 'x', 'y', 'z', 'radius', 'parent']
    df = pd.DataFrame(data_array, columns=colnames)
    return df[1:]
def distance(i,j, treedf):
    # eculidean distance of two nodes
    import numpy as np
    df = treedf
    coord1 = np.array([df['x'][i], df['y'][i], df['z'][i]])
    coord2 = np.array([df['x'][j], df['y'][j], df['z'][j]])
    dist = np.linalg.norm(coord1-coord2)
    return dist
def convert2nx(df):
    # graph: python dict
    import networkx as nx
    gi = nx.Graph()
    n = len(df)
    for i in range(1,n+1): # change later
        gi.add_node(i, coordinates = (df['x'][i], df['y'][i], df['z'][i]), structure = df['structure'][i], radius = df['radius'][i])
        if df['parent'][i] != -1:
            gi.add_edge(int(df['id'][i]), int(df['parent'][i]), length = distance(i, df['parent'][i], df))
    gi.edges
    gi.nodes[1]
    assert nx.is_tree(gi)
    # assert gi.nodes[1]['coordinates'] == (0, 0, 0)
    return gi
def function_basis(g):
    # input: g
    # output: g with ricci, deg, hop, cc, fiedler computed
    import networkx as nx
    import sys
    import numpy as np
    assert nx.is_connected(g)
    sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/')
    from GraphRicciCurvature.OllivierRicci import ricciCurvature
    def norm(g, key):
        # return 0.01
        # get the max of g.node[i][key]
        for v, data in sorted(g.nodes(data=True), key=lambda x: abs(x[1][key]), reverse=True):
            norm = np.float(data[key])
            return norm

    g_ricci = g
    # degree
    deg_dict = dict(nx.degree(g))
    deg_norm = np.float(max(deg_dict.values()))
    for n in g_ricci.nodes():
        g_ricci.node[n]['deg'] = deg_dict[n]

    # hop
    dist_dict = nx.single_source_dijkstra_path_length(g_ricci, 1, weight='length')
    for n in g_ricci.nodes():
        g_ricci.node[n]['dist'] = dist_dict[n]
        g_ricci.node[n]['direct_distance'] = np.linalg.norm(np.array(g.nodes[n]['coordinates']) - np.array(g.nodes[1]['coordinates']) )


    # fiedler
    # from networkx.linalg.algebraicconnectivity import fiedler_vector
    # fiedler = fiedler_vector(g, normalized=False)  # np.ndarray
    # assert max(fiedler) > 0
    # fiedler = fiedler / max(fiedler)
    # assert max(fiedler) == 1
    # for n in g_ricci.nodes():
    #     n_idx = list(g_ricci.nodes).index(n)
    #     g_ricci.node[n]['fiedler'] = fiedler[n_idx]

    return g_ricci
def print_node_vals(g, key):
    n = len(g)
    for i in range(1, n+1):
        print(i, g.nodes[i][key])
# print_node_vals(g, 'dist')

def find_node_val(g, key, val):
    import numpy as np
    n = len(g); flag = 0
    for i in range(1, n + 1):
        if np.abs(g.nodes[i][key] - val) < 0.01:
            print(i, g.nodes[i][key]);
            flag = 1
    if flag == 0:
        print('Did not match')
def get_diagram(g, key='dist', typ='tuple', subflag = 'False'):
    # only return 0-homology of sublevel filtration
    # type can be tuple or pd. tuple can be parallized, pd cannot.
    import dionysus as d
    def get_simplices(gi, key=key):
        assert str(type(gi)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
        import networkx as nx
        assert len(list(gi.node)) > 0
        assert key in gi.node[list(gi.nodes)[2]].keys()

        simplices = list()
        for u, v, data in sorted(gi.edges(data=True), key=lambda x: x[2]['length']):
            # tup = ([u, v], data['length'])
            tup = ([u, v], min(gi.nodes[u][key], gi.nodes[v][key]))
            simplices.append(tup)

        for v, data in sorted(gi.nodes(data=True), key=lambda x: x[1][key]):
            tup = ([v], data[key])
            simplices.append(tup)

        return simplices

    simplices = get_simplices(g)

    def compute_PD(simplices, sub=True, inf_flag='False'):
        import dionysus as d
        f = d.Filtration()
        for simplex, time in simplices:
            f.append(d.Simplex(simplex, time))
        if sub == True:
            f.sort()
        elif sub == False:
            f.sort(reverse=True)
        for s in f:
            continue
            print(s)
        m = d.homology_persistence(f)
        # for i,c in enumerate(m):
        #     print(i,c)
        dgms = d.init_diagrams(m, f)
        # print(dgms)
        for i, dgm in enumerate(dgms):
            for pt in dgm:
                continue
                print(i, pt.birth, pt.death)

        def del_inf(dgms):
            import dionysus as d
            dgms_list = [[], []]
            for i in range(2):
                pt_list = list()
                for pt in dgms[i]:
                    if (pt.birth == float('inf')) or (pt.death == float('inf')):
                        pass
                    else:
                        pt_list.append(tuple([pt.birth, pt.death]))
                diagram = d.Diagram(pt_list)
                dgms_list[i] = diagram

            return dgms_list

        if inf_flag == 'False':
            dgms = del_inf(dgms)

        return dgms

    super_dgms = compute_PD(simplices, sub=False)
    sub_dgms = compute_PD(simplices, sub=True)
    n_node = len(g.nodes)
    _min = min([g.nodes[n][key] for n in g.nodes])
    _max = max([g.nodes[n][key] for n in g.nodes])+ 1e-5 # avoid the extra node lies on diagonal
    p_min = d.Diagram([(_min, _max)])
    p_max = d.Diagram([(_max, _min)])

    # print(p[0])
    sub_dgms[0].append(p_min[0])
    super_dgms[0].append(p_max[0])
    # add super_level filtration
    # for p in super_dgms[0]:
    #     sub_dgms[0].append(p)
    if subflag=='True':
        return sub_dgms[0]
    elif subflag=='False':
        return super_dgms[0]
    # no longer needed since mrzv has fix the error
    # tuple_dgms = [(p.birth, p.death) for p in sub_dgms[0]]

    # return sub_dgms[0]
    # if typ == 'tuple':
    #     return tuple_dgms

def number_lines(i):
    file_directory = files[i - 1]
    count = len(open(file_directory).readlines())
    return count
def count_lines():
    for i in range(1, 1000):
        print(i,  number_lines(i))
def convert2dayu(g):
    direct = '/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/NeuronTools/Test/dayu.txt'
    f= open(direct, 'w')
    f.write(str(len(g)) + '\n' )
    for i in g.nodes():
        nval = g.nodes[i]['direct_distance']
        f.write(str(nval) + '\n')
    for e in g.edges():
        f.write(str(e[0]-1) + ' ' + str(e[1]-1) + '\n' )
        # print e[0],
        # print e[1]
def export_dgm(i, dgm, files, key):
    file1 = '/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/NeuronTools/TruePDs/data_1268/'
    file2 = '/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/TruePDs/data_1268/'
    neuron_id = key + '/' + str(i) + '_' + files[i-1].split('/')[-1]
    try:
        f = open(file1 + neuron_id, 'w+')
    except:
        f = open(file2 + neuron_id, 'w+')
    diag = dgm2diag(dgm)
    for pd in diag:
        f.write(str(pd[0]) + ' ' + str(pd[1]) + '\n')
    f.close()

# for i in range(1, 1269):
#     id = i
#     # neuronPD = read_neuronPD(id)
#     # Neuron_dgm = diag2dgm(neuronPD)
#     # print_dgm(Neuron_dgm)
#     files = get_swc_files()
#     df = get_df(files, id)
#     tree = convert2nx(df)
#     g = function_basis(tree)
#     import networkx as nx
#     nx.is_tree(g)
#     # convert2dayu(g)
#     dgm = get_diagram(g, key='direct_distance')
#     export_dgm(id, dgm, files)
#     print('Finish tree %s'%i)

def computePD(i, key='direct_distance'):
    id = i
    files = get_swc_files()
    df = get_df(files, id)
    tree = convert2nx(df)
    g = function_basis(tree)
    import networkx as nx
    nx.is_tree(g)
    # convert2dayu(g)
    dgm = get_diagram(g, key=key)
    export_dgm(id, dgm, files)
    print('Finish tree %s' % i)

from joblib import delayed, Parallel
Parallel(n_jobs=-1)(delayed(computePD)(i, 'dist') for i in range(1, 1269))


def check_functionval(g, Neuron_dgm):
    vals = dgm_distinct(Neuron_dgm)
    for val in vals:
        find_node_val(g, 'direct_distance', val)
def scatter_comp():
    import matplotlib.pyplot as plt
    import numpy as np
    data = dgm2diag(Neuron_dgm)
    x = np.array([data[i][0] for i in range(len(data)-1)])
    y = np.array([data[i][1] for i in range(len(data)-1)])
    plt.scatter(x, y, c='b')
    plt.show()