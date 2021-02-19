import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

signal_size = 1024


datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
normalname = ["97.mat", "98.mat", "99.mat", "100.mat"]
# For 12k Drive End Bearing Fault Data
dataname1 =  [ "169.mat","185.mat", "197.mat"]  # 1797rpm 69 85 97
dataname2 = ["170.mat", "186.mat", "198.mat"]  # 1772rpm
"""dataname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
             "236.mat"]  # 1750rpm
dataname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
             "237.mat"]  # 1730rpm"""

dataname5 =  ["274.mat", "286.mat", "310.mat"]  # 1797rpm
dataname6 = ["275.mat", "287.mat", "309.mat"] # 1772rpm
"""dataname7 = ["280.mat", "284.mat", "296.mat", "276.mat", "288.mat", "311.mat", "272.mat", "292.mat",
             "317.mat"]  # 1750rpm
dataname8 = ["281.mat", "285.mat", "297.mat", "277.mat", "289.mat", "312.mat", "273.mat", "293.mat",
             "318.mat"]  # 1730rpm"""
# For 48k Drive End Bearing Fault Data
dataname9 = ["173.mat", "189.mat", "201.mat"]  # 1797rpm
dataname10 = ["175.mat", "190.mat", "202.mat"] # 1772rpm
"""dataname11 = ["111.mat", "124.mat", "137.mat", "176.mat", "191.mat", "203.mat", "215.mat", "228.mat",
              "240.mat"]  # 1750rpm
dataname12 = ["112.mat", "125.mat", "138.mat", "177.mat", "192.mat", "204.mat", "217.mat", "229.mat",
              "241.mat"]  # 1730rpm"""

target_data= {0:["NA.mat","IF.mat","BF.mat","OF.mat"]}  #NIBO
label2= [i for i in range(0, 4)]



# label
label = [1, 2, 3]  # The failure data is labeled 1-9
axis = ["_DE_time", "_FE_time", "_BA_time"]


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data_root1 = os.path.join('/Users/macbookair/PycharmProjects/Shad',root, datasetname[3])
    data_root2 = os.path.join('/Users/macbookair/PycharmProjects/Shad',root, datasetname[0])
    data_root3 = os.path.join('/Users/macbookair/PycharmProjects/Shad', root, datasetname[1])
    data_root4 = os.path.join('/Users/macbookair/PycharmProjects/Shad', root, datasetname[2])


    path1 = os.path.join('/Users/macbookair/PycharmProjects/Shad',data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    dataM, labM = data_load(path1, axisname=normalname[0],label=0)  # nThe label for normal data is 0
    print("Len of data",len(dataM))


    """path1 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root1,
                         normalname[1])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    dataN, labN = data_load(path1, axisname=normalname[1], label=0)  # nThe label for normal data is 0
    print("Len of data", len(dataN))

    path1 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root1,
                         normalname[2])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    dataO, labO = data_load(path1, axisname=normalname[2], label=0)  # nThe label for normal data is 0
    print("Len of data", len(dataO))

    path1 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root1,
                         normalname[3])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    dataP, labP = data_load(path1, axisname=normalname[3], label=0)  # nThe label for normal data is 0
    print("Len of data", len(dataP))"""


    dataMN = dataM#+dataN
    labMN= labM#+labN


    dataMNO= dataMN#+dataO
    labMNO = labMN#+labO

    data = dataMNO#+dataP
    lab = labMNO#+ labP




    for i in tqdm(range(len(dataname1))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad',data_root2, dataname1[i])

        data1, lab1 = data_load(path2, dataname1[i], label=label[i])
        #print("Len of dat1",len(data1))
        data += data1
        lab += lab1

    print("After the first add" ,len(lab))
    """for i in tqdm(range(len(dataname2))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root2, dataname2[i])

        data1, lab1 = data_load(path2, dataname2[i], label=label[i])
        data += data1
        lab += lab1

    print("After the second add",len(lab))

    for i in tqdm(range(len(dataname3))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root2, dataname3[i])

        data1, lab1 = data_load(path2, dataname3[i], label=label[i])
        data += data1
        lab += lab1

    print("After the third add",len(lab))

    for i in tqdm(range(len(dataname4))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root2, dataname4[i])

        data1, lab1 = data_load(path2, dataname4[i], label=label[i])
        data += data1
        lab += lab1

    print("After the 4th add",len(lab))

    for i in tqdm(range(len(dataname5))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root3, dataname5[i])

        data1, lab1 = data_load(path2, dataname5[i], label=label[i])
        data += data1
        lab += lab1

    print("After the 5th add",len(lab))

    for i in tqdm(range(len(dataname6))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root3, dataname6[i])

        data1, lab1 = data_load(path2, dataname6[i], label=label[i])
        data += data1
        lab += lab1

    print("After the 6th add",len(lab))

    for i in tqdm(range(len(dataname7))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root3, dataname7[i])

        data1, lab1 = data_load(path2, dataname7[i], label=label[i])
        data += data1
        lab += lab1

    print("After the 7th add",len(lab))

    for i in tqdm(range(len(dataname8))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root3, dataname8[i])

        data1, lab1 = data_load(path2, dataname8[i], label=label[i])
        data += data1
        lab += lab1

    print("After the 8th add",len(lab))

    for i in tqdm(range(len(dataname9))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root4, dataname9[i])

        data1, lab1 = data_load(path2, dataname9[i], label=label[i])
        data += data1
        lab += lab1

    print("After the 9th add",len(lab))

    for i in tqdm(range(len(dataname10))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root4, dataname10[i])

        data1, lab1 = data_load(path2, dataname10[i], label=label[i])
        data += data1
        lab += lab1

    print("After the 10th add",len(lab))

    for i in tqdm(range(len(dataname11))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root4, dataname11[i])

        data1, lab1 = data_load(path2, dataname11[i], label=label[i])
        data += data1
        lab += lab1

    print("After the 11th add",len(lab))

    for i in tqdm(range(len(dataname12))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root4, dataname12[i])

        data1, lab1 = data_load(path2, dataname12[i], label=label[i])
        data += data1
        lab += lab1

    print("After the 12th add",len(lab))"""



    return [data, lab]



def get_files_t(root,N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(target_data[N[k]]))):
            if n == 0:
                path1 = os.path.join(root, target_data[N[k]][n])
            else:
                path1 = os.path.join(root, target_data[N[k]][n])
            data1, lab1 = data_load_t(path1, label=label2[n])
            data += data1
            lab += lab1

    # print('step1')
    # print(lab)
    return [data, lab]




def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    #print("before if datanumber",datanumber)
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
        #print("If data number",datanumber)
    else:
        realaxis = "X" + datanumber[0] + axis[0]
        #print("else data num",datanumber)
    fl = loadmat(filename)[realaxis]
    #print("file name",filename)
    data = []
    lab = []
    start, end = 0, signal_size


    while end <= fl.shape[0]:
        x = fl[start:end]
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1, 1)
        data.append(x)
        lab.append(label)
        start += signal_size
        end += signal_size
    #print("After while datalen and lablen",len(data),len(lab))
    return data, lab


def data_load_t(filename,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''


    fl = loadmat(filename)['data']
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        x = fl[start:end]
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1, 1)
        #print(len(x))
        data.append(x)
        lab.append(label)
        start += signal_size
        end += signal_size

    print(filename)
    print("A",lab)
    return data, lab



class CWRU(object):
    num_classes = 4
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="mean-std"):
        self.data_dir = data_dir
        #self.source_N = transfer_task[0]
        #print(self.source_N)
        #self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                RandomAddGaussian(),
                RandomScale(),
                RandomStretch(),
                RandomCrop(),
                Retype(),
                #Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                #Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files_t(self.data_dir,[0])
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            #data_pd2 = pd.DataFrame({"data": list_data[0]})


            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=train_pd, transform=self.data_transforms['val'])
            print("Hello",len(source_train),len(source_val),len(target_train),len(target_val))
            #target_train = list(list_data[0])
            return source_train, source_val, target_train, target_val
        """else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            print("source train",len(source_train),len(source_val),len(target_val))

            return source_train, source_val, target_val"""



"""
    def data_split(self):

"""