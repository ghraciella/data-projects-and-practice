import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


batch_size = 64
t_batch_size= 1000
n_train_samples = 32000
n_val_samples = 10000

path = 'C:/Users/eze/Documents/PhD_project/datasets/digit-recognizer'

data_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))
])

labels_transform = transforms.Compose([transforms.ToTensor()])



class CustomMNISTDataset(Dataset):
    def __init__(self, path= path, data_transforms=None, label_transforms=None, mode = 'train', is_labels = True, *args, **kwargs):
        
        self.dataset = pd.read_csv(path)
        self.mode = mode
        self.is_labels = is_labels
        self.height, self.width = 28, 28
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms
         
    def __len__(self):
        return (len(self.dataset))
    
    def __getitem__(self, index):

        if self.mode == 'train':
            data, labels = self.dataset.iloc[index, 1:], self.dataset.iloc[index, 0]


        elif self.mode == 'test':
            data = self.dataset.iloc[index, :]
            labels = None

        images = data 
        images = np.asarray(images).astype(np.uint8).reshape(self.height, self.width, 1)

        label = labels 
        
        if self.data_transforms is not None:
           images = self.data_transforms(images)
        if self.label_transforms is not None:
           label = self.label_transforms(labels)
   
        if self.is_labels == True:
           return (images, label)
        else:
           return images



model_type = 'node'
train_data = CustomMNISTDataset(path= path + '/train.csv', data_transforms=data_transform, label_transforms=None, mode = 'train')
test_data = CustomMNISTDataset(path= path + '/test.csv', data_transforms=data_transform, label_transforms=None, mode = 'test', is_labels = False)




def CDataloader(train_data=train_data, test_data=test_data, model_type=model_type, batch_size=batch_size):

    train_subset, val_subset = torch.utils.data.random_split(train_data, [n_train_samples, n_val_samples], generator=torch.Generator().manual_seed(1))

    if model_type == 'standard':
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if model_type == 'node':
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)


    return trainloader, valloader, testloader









if __name__ == "__main__": 


    
    epochs = 5
    learning_rate = 1e-3 #1e-4
    batch_size = 64
    t_batch_size= 1000
    n_train_samples = 32000
    n_val_samples = 10000
    drop_ode = False
    model_type =  'node' #'standard' 

    #path = 'C:/Users/eze/Documents/PhD_project/datasets/digit-recognizer'

    train_data = CustomMNISTDataset(path= path + '/train.csv', data_transforms=data_transform, label_transforms=None, mode = 'train')
    test_data = CustomMNISTDataset(path= path + '/test.csv', data_transforms=data_transform, label_transforms=None, mode = 'test', is_labels = False)


    img, lbl = train_data.__getitem__(0)
    print(img.size())

    timg = test_data.__getitem__(0)
    print(timg.size())

    trlen, telen = train_data.__len__(), test_data.__len__()
    print(f"train dataset lenght is {trlen} , test dataset lenght is {telen}")


    #train_subset, val_subset = torch.utils.data.random_split(train_data, [32000, 10000], generator=torch.Generator().manual_seed(1))

    #trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    #valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    #testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)



    trainloader, valloader, testloader = CDataloader(train_data=train_data, test_data=test_data, model_type=model_type, batch_size=batch_size)


    train_iter = enumerate(trainloader)
    batch_idx, (train_imgdata, train_imglabels) = next(train_iter)
    print('train data shape: ', train_imgdata.shape)


    val_iter = enumerate(valloader)
    batch_jdx, (val_imgdata, val_imglabels) = next(val_iter)
    print('val data shape: ', val_imgdata.shape)




















