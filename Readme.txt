
The code is used to classify IMS dataset only using CWRU and labels Using Domain Adversarial Neural Network and Transfer Learning. 

Use the following command to run the code. 
Change the data directory to the path of the dataset.

train_advanced.py is the main file. Change the arguments in it to customise the code.


python train_advanced.py --data_name CWRU --data_dir data/Mechanical-datasets --transfer_task [0],[1]  --domain_adversarial True --adversarial_loss DA

Save models folder contains the models and train log which achieves 85 percent accuracy.
Models are stored with epoch number and accuracy as the file name.


Please cite the code owner if ever used in other premises.
Shadrach Simon Sundar
M.Eng Deggendorf Institute of Technology