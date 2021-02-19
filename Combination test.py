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
dataname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
             "234.mat"]  # 1797rpm
dataname2 = ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
             "235.mat"]  # 1772rpm
dataname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
             "236.mat"]  # 1750rpm
dataname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
             "237.mat"]  # 1730rpm



target_data= {0:["NA.mat","IF.mat","BF.mat","OF.mat"]}
label2= [i for i in range(0, 4)]



# label
label = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # The failure data is labeled 1-9
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

    path1 = os.path.join('/Users/macbookair/PycharmProjects/Shad',data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    dataM, labM = data_load(path1, axisname=normalname[0],label=0)  # nThe label for normal data is 0
    print("Len of data",len(dataM))


    path1 = os.path.join('/Users/macbookair/PycharmProjects/Shad', data_root1,
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
    print("Len of data", len(dataP))


    dataMN = dataM+dataN
    labMN= labM+labN


    dataMNO= dataMN+dataO
    labMNO = labMN+labO

    data = dataMNO+dataP
    lab = labMNO+ labP


    print("data", (data))

    for i in tqdm(range(len(dataname1))):
        path2 = os.path.join('/Users/macbookair/PycharmProjects/Shad',data_root2, dataname1[i])

        data1, lab1 = data_load(path2, dataname1[i], label=label[i])
        #print("Len of dat1",len(data1))
        data += data1
        lab += lab1

    print("After the first add" ,len(lab))
    for i in tqdm(range(len(dataname2))):
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
        data.append(fl[start:end])
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
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
    print(filename)
    print(lab)
    return data, lab



class CWRU(object):
    num_classes = 10
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        #self.source_N = transfer_task[0]
        #print(self.source_N)
        #self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
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
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
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