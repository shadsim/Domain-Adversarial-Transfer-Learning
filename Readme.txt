
The code is used to classify IMS dataset only using CWRU and labels Using Domain Adversarial Neural Network and Transfer Learning. 

Use the following command to run the code. 
Change the data directory to the path of the dataset.

train_advanced.py is the main file. Change the arguments in it to customise the code.


python train_advanced.py --data_name CWRU --data_dir data/Mechanical-datasets --transfer_task [0],[1]  --domain_adversarial True --adversarial_loss DA

Save models folder contains the models and train log which achieves 85 percent accuracy.
Models are stored with epoch number and accuracy as the file name.


Requirements

Python 3.7
Numpy 1.16.2
Pandas 0.24.2
Pickle
tqdm 4.31.1
sklearn 0.21.3
Scipy 1.2.1
opencv-python 4.1.0.25
PyWavelets 1.0.2
pytorch >= 1.1
torchvision >= 0.40


Please cite the code owner if ever used in other premises.
Shadrach Simon Sundar
M.Eng Deggendorf Institute of Technology
