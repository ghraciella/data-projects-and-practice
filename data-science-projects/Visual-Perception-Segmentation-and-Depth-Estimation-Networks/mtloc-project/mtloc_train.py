import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import copy
import time
from mtloc_load_dataset import *
from mtloc_models import *
from mtloc_optimizers import *
from mtloc_objective_losses import *
from mtloc_visualise import *
from mtloc_node import *

#from ada_hessian import AdaHessian

#time processin/training time

torch.manual_seed(0)



#input_features = 784
drop_ode = False
epochs = 30





def get_accuracy(predictions, true_labels):
    _, predicted = torch.max(predictions, 1)
    corrects = (predicted == true_labels).sum()
    accuracy = 100.0 * corrects/len(true_labels)
    return accuracy.item()




def training(trainloader,  epochs, model, loss_fn, optimizer, drop_ode=drop_ode, model_type=model_type, scheduler=False, hessian=False, *args, **kwargs):

    train_accuracies, train_losses = [], []
    
    # set the train mode
    model.train()

    for epoch in range(epochs):        

        train_loss = 0 
        train_accuracy = 0
        num_batch = 0

        
        for i, (train_data, train_labels) in enumerate(trainloader):

            optimizer.zero_grad()
            
            if model_type == 'node':
                if not drop_ode:
                    #train_data = train_data.view(batch_size, -1)
                    train_data = train_data.view(train_data.size(0), -1)
                    train_preds = model(train_data, drop_ode=drop_ode)
                    loss = F.nll_loss(train_preds, train_labels)
                    #print(train_preds.shape, train_labels.shape)

            if model_type=='standard':
                train_preds = model(train_data)
                loss = loss_fn(train_preds, train_labels)

            
            if i % 100 == 0:
                print(f'[Training] {i}/{epoch}/{epochs} -> Loss: {loss.item()}')
                
            train_acc = get_accuracy(train_preds, train_labels)
            
            #print('weight parameters before backward pass: \n', model.fc1.weight.grad)

            loss.backward()

            if hessian==True:
                loss.backward(create_graph=True)  # if hessian, create graph

            #print('weight parameters after backward pass: \n', model.fc2.weight.grad)
              
            optimizer.step()       
            
            if scheduler == True:
                scheduler.step(loss)

            num_batch += 1
            train_loss += loss.item()
            train_accuracy += train_acc


    
        epoch_accuracy = train_accuracy/num_batch
        epoch_loss = train_loss/num_batch        
        train_accuracies.append(epoch_accuracy)
        train_losses.append(epoch_loss)
        
        print("Epoch: {}/{} ".format(epoch + 1, epochs),
              "Train Loss: {:.4f} ".format(epoch_loss),
              "Train accuracy: {:.4f}".format(epoch_accuracy))
    
    return train_accuracies, train_losses




def main_test():




    width, height = 28, 28
    input_features = width*height
    epochs = 150
    momentum = 0.9
    alpha = 0.99
    #beta1, beta2 = 0.99, 0.999
    learning_rate = 1e-3 #1e-4
    batch_size = 128 # 64
    t_batch_size= 1000
    n_train_samples = 32000
    n_val_samples = 10000
    drop_ode = False
    #model_type = 'node'
    model_type = 'standard'


    path = 'C:/Users/eze/Documents/PhD_project/datasets/digit-recognizer'

    path = 'C:/Users/eze/Documents/PhD_project/datasets/digit-recognizer'


    train_data = CustomMNISTDataset(path= path + '/train.csv', data_transforms=data_transform, label_transforms=None, mode = 'train')
    test_data = CustomMNISTDataset(path= path + '/test.csv', data_transforms=data_transform, label_transforms=None, mode = 'test', is_labels = False)


    trainloader, valloader, testloader = CDataloader(train_data=train_data, test_data=test_data, model_type=model_type, batch_size=batch_size)


    #mnist = torchvision.datasets.MNIST('./mnist', download=True, train=True, transform=torchvision.transforms.ToTensor())
    #mnisttest = torchvision.datasets.MNIST('./mnist', download=True, train=False, transform=torchvision.transforms.ToTensor())
    #mnistdl = torch.utils.data.DataLoader(mnist, shuffle=True, batch_size=batch_size, drop_last=True, pin_memory=True)
    #mnisttestdl = torch.utils.data.DataLoader(mnisttest, shuffle=True, batch_size=batch_size, drop_last=True, pin_memory=True)

    #CIFAR100
    cifar10 = torchvision.datasets.CIFAR10('./cifar10', download=True, train=True, transform=transforms.Compose(
        [transforms.Resize([28, 28]),
         transforms.Grayscale(num_output_channels = 1),
         transforms.ToTensor()]))
    cifar10test = torchvision.datasets.CIFAR10('./cifar10', download=True, train=False, transform=transforms.Compose(
        [transforms.Resize([28, 28]),
         transforms.ToTensor()]))
    cifar10dl = torch.utils.data.DataLoader(cifar10, shuffle=True, batch_size=batch_size, drop_last=True, pin_memory=True)
    cifar10dl = torch.utils.data.DataLoader(cifar10, shuffle=True, batch_size=batch_size, drop_last=True, pin_memory=True)


    #for e in range(epochs):
    #    for i, (predictions, labels) in enumerate(mnistdl):
    #        print('gen mnist preds shape', predictions.shape)
    #        print('gen mnist labels shape', labels.shape)


    #for e in range(epochs):
    #    for batch_idx, (mpredictions, mlabels) in enumerate(trainloader):
    #        print('my mnist', mpredictions.shape)



    ##initialize model

    if model_type == 'standard':
        model =simple_model(input_features, 250, 10,  'relu')
        print('Forward propagation:', model)
        summary(model, (1, width, height))

    if model_type == 'node':
        ###Neural ODE model
        dynamics = Dynamics(28 * 28)
        model = ODEClassifier(28 * 28, 10, dynamics)

    if torch.cuda.is_available():
        model = model.cuda()



    ##Optimization

    #loss
    loss_fn = nn.CrossEntropyLoss()

    #optimizer
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=momentum, weight_decay=1e-4)
    optimizer_rms = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
    optimizer_A = torch.optim.Adam(model.parameters(), lr = learning_rate)
    optimizer_Ah = AdaHessian(model.parameters(), lr = learning_rate)

    optimizer = optimizer_A

    # verbose=True will print learning rate changes
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, threshold=0.0001, patience=3, verbose=True)

    #train_accuracies, train_losses = training(trainloader, epochs, model, loss_fn, optimizer)#, scheduler = False , hessian=False)

    train_accuracies, train_losses = training(cifar10dl, epochs, model, loss_fn, optimizer, drop_ode=drop_ode, model_type=model_type)

    print("begin visualization............check out your photos :)")

    train_curves(epochs, train_losses, train_accuracies)

    print("cool photos, okay bye!")






if __name__ == "__main__": 


    epochs = 5
    momentum = 0.9
    alpha = 0.99
    #beta1, beta2 = 0.99, 0.999
    learning_rate = 1e-3 #1e-3
    batch_size = 64
    t_batch_size= 1000
    n_train_samples = 32000
    n_val_samples = 10000
    drop_ode = False
    model_type = 'node'

    path = 'C:/Users/eze/Documents/PhD_project/datasets/digit-recognizer'


    train_data = CustomMNISTDataset(path= path + '/train.csv', data_transforms=data_transform, label_transforms=None, mode = 'train')
    test_data = CustomMNISTDataset(path= path + '/test.csv', data_transforms=data_transform, label_transforms=None, mode = 'test', is_labels = False)


    trainloader, valloader, testloader = CDataloader(train_data=train_data, test_data=test_data, model_type=model_type, batch_size=batch_size)


    
    print("Training in progress!!")
    print("---------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------")

    #optimizer
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=momentum, weight_decay=1e-4)
    optimizer_rms = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
    optimizer_A = torch.optim.Adam(model.parameters(), lr = learning_rate)
    optimizer_Ah = AdaHessian(model.parameters(), lr = learning_rate)

    optimizer = optimizer_A
    train_accuracies, train_losses = training(trainloader,  epochs, model, loss_fn, optimizer, drop_ode=True)

    print("End of training!!")


    main_test()



























