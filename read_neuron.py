file = '/Users/admin/Documents/osu/Research/DeepGraphKernels/linux_sync/deep-persistence/NeuronTools/persistence_diagrams/1_data_1268/1000_MTC080300A-IDA.CNG.swc'
f = open(file, 'r')
pd = f.readlines()

def read_data(id=3, type="ricci", hom=0, filtration='sub'):
    """
    Read data from NeuronTools
    :return: a list of tuples
    """
    file_directory = '/Users/admin/Documents/osu/Research/DeepGraphKernels/linux_sync/deep-persistence/NeuronTools/persistence_diagrams/1_data_1268/1000_MTC080300A-IDA.CNG.swc'
    # file_directory ="/Users/admin/Documents/osu/Research/DeepGraphKernels/datasets/enzyme/prefiltration/"
    # file_directory = file_directory + type + "/" + str(id) + homology + filtration+ "PD" + ".txt"
    file = open(file_directory,"r")
    data = file.read(); data = data.split('\r\n');
    Data = []
    # this may need to change for different python version
    for i in range(0,len(data)-1):
        data_i = data[i].split(' ')
        data_i_tuple = (float(data_i[0]), float(data_i[1]))
        Data = Data + [data_i_tuple]
    return Data

