import PC
import proxPC
import torch
import torchvision
import torch.optim as optim
import utilities
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
import copy
import math

bce = torch.nn.BCELoss(reduction='none')
mse = torch.nn.MSELoss(reduction='none')
softmax = torch.nn.Softmax(dim=1)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
relu = torch.nn.ReLU()


# Load MNIST Data
def get_data(batch_size=64, data=0):
    if data == 2:
        d_name = 'CIFAR'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                  shuffle=False)



    elif data == 1:
        num_train = 60000
        d_name = 'fashion'

        train_loader = DataLoader(
            torchvision.datasets.FashionMNIST('./data', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                              ])), batch_size=batch_size, shuffle=True, pin_memory=False)

        test_loader = DataLoader(
            torchvision.datasets.FashionMNIST('./data', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                              ])), batch_size=10000, shuffle=False, pin_memory=False)
    else:
        num_train = 60000
        d_name = ''

        train_loader = DataLoader(
            torchvision.datasets.MNIST('./data', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                       ])), batch_size=batch_size, shuffle=True, pin_memory=False)

        test_loader = DataLoader(
            torchvision.datasets.MNIST('./data', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                       ])), batch_size=10000, shuffle=False, pin_memory=False)

    return train_loader, test_loader, d_name, num_train



def compute_means(data):
    with torch.no_grad():
        d_tensor = torch.tensor(data[0]).view(1, -1)
        for m in range(1, len(data)):
            d_tensor = torch.cat((d_tensor, torch.tensor(data[m]).view(1, -1)), dim=0)
        return torch.mean(d_tensor, dim=0)



def test(test_losses, test_accuracies, model, test_loader, seed, lr, dev):
    with torch.no_grad():
        for batch_idx, (images, y) in enumerate(test_loader):

            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)

            # Transform targets, y, to onehot vector
            target = torch.zeros(images.size(0), 10, device=dev)
            utilities.to_one_hot(target, y, dev=dev)

            # Test and record losses and accuracy over whole test set
            h = model.compute_values(images)
            global_loss = torch.mean(mse(softmax(h[-1]), target).sum(1))
            test_accuracies[lr][seed].append(utilities.compute_num_correct(softmax(h[-1]), y) / images.size(0))
            test_losses[lr][seed].append(global_loss.item())



def train_model(train_loader, test_loader, model, seed, lr, test_losses, test_accuracies, wt_norms, max_iters, dev):
    iter = 0
    test(test_losses, test_accuracies, model, test_loader, seed, lr, dev)

    while iter < max_iters:
        for batch_idx, (images, y) in enumerate(train_loader):
            images = images.view(y.size(0), -1).to(dev)

            y = y.to(dev)

            # Transform targets, y, to onehot vector
            target = torch.zeros(images.size(0), 10, device=dev)
            utilities.to_one_hot(target, y, dev=dev)

            _, _, _, dw = model.train_wts(images.detach(), target.detach(), y.detach())

            # Test every 50 training iterations
            iter += 1
            if iter % 50 == 0 and iter > 0:
                test(test_losses, test_accuracies, model, test_loader, seed, lr, dev)
                wt_norms[lr][seed].append(torch.norm(dw).item())
                #print(iter, test_accuracies[lr][seed][-1])
                '''if iter % 800 == 0:
                    print('Epoch', iter/781.25, test_accuracies[lr][seed][-1])'''

            # End if max training iterations reached
            if iter == max_iters:
                return






def train(models, batch_size, data, dev, max_iters, test_losses, test_accuracies, wt_norms, lrs):

    for l in range(len(models)):
        print(f'Training LR:{lrs[l]}')
        for m in range(len(models[0])):

            train_loader, test_loader, d_name, num_train = get_data(batch_size, data=data)

            train_model(train_loader, test_loader, models[l][m], m, l, test_losses, test_accuracies, wt_norms, max_iters, dev)

            print(f'Seed:{m} Acc:', test_accuracies[l][m][-1])







def training_run(max_iters=10000, batch_size=1, data=0, num_seeds=1, n=1, smax=True, lrs=[.015], model_type=0,
                 save_best=True, eps=.25):

    # Create Models
    if data == 2:
        model_dim = [3072, 1024, 1024, 1024, 10]
    else:
        model_dim = [784, 256, 256, 10]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []

    for l in range(len(lrs)):
        #Add list of seeds at this learning rate
        models.append([])
        for m in range(num_seeds):

            #BP-SGD
            if model_type == 0:
                models[-1].append(PC.PC(model_dim, t_type=0, lr=lrs[l], n=n, crossEnt=True, adam=False, smax=smax))
            #BP-Adam
            elif model_type == 1:
                models[-1].append(PC.PC(model_dim, t_type=0, lr=lrs[l], n=n, crossEnt=True, adam=True, smax=smax))

            #IL-SGD
            elif model_type == 3:
                models[-1].append(PC.PC(model_dim, t_type=2, lr=lrs[l], n=n, crossEnt=True, adam=False, smax=smax,
                                        n_iter=25, bot_gamma=.02, top_gamma=.015))

            #IL-prox
            elif model_type == 6:
                models[-1].append(proxPC.ProxPC(model_dim, lr=lrs[l], crossEnt=True, mom=False, smax=smax, n_iter=25,
                                    bot_gamma=.015, top_gamma=.015, dec_gamma=.0, normalize=True, eps=eps))

            #IL-prox Fast
            elif model_type == 7:
                models[-1].append(proxPC.ProxPC(model_dim, lr=lrs[l], crossEnt=True, mom=False, smax=smax, n_iter=12,
                                    bot_gamma=.025, top_gamma=.0, dec_gamma=.0, normalize=True, eps=eps))

            #BP-prox
            elif model_type == 8:
                models[-1].append(proxPC.ProxPC(model_dim, lr=lrs[l], crossEnt=True, mom=False, smax=smax, t_type=3, eps=eps))

            #IL-Adam
            elif model_type == 9:
                models[-1].append(PC.PC(model_dim, t_type=2, lr=lrs[l], n=n, crossEnt=True, adam=True, smax=smax,
                                        n_iter=20, bot_gamma=.02, top_gamma=.01))

            #IL-prox Adam
            elif model_type == 10:
                models[-1].append(proxPC.ProxPC(model_dim, lr=lrs[l], crossEnt=True, smax=smax, n_iter=20,
                                            bot_gamma=.02, top_gamma=.015, dec_gamma=.0, normalize=True, eps=eps, mom=True))


        # To Device
        for i in range(len(models[-1])):
            models[-1][i].to(dev)

    #################################################
    # Create Containers
    test_losses = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]
    test_accuracies = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]
    wt_norms = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]

    #################################################

    # Train
    print(f'TRAINING MODEL TYPE {model_type}')
    train(models, batch_size, data, dev, max_iters, test_losses, test_accuracies, wt_norms, lrs)


    if save_best:
        # Store Data
        best_test_acc = torch.mean(torch.tensor([test_accuracies[0][x][-1] for x in range(len(test_accuracies[0]))])).item()
        best_lr = 0
        for l in range(1, len(models)):
            ac = torch.mean(torch.tensor([test_accuracies[l][x][-1] for x in range(len(test_accuracies[0]))])).item()
            if best_test_acc < ac:
                best_test_acc = ac
                best_lr = l

        print(f'Best Learning Rate, at Iterations{max_iters}, Model Type{model_type}:', best_lr)
        with open(f'data/BestLrRun_BP_modelType{model_type}_data{data}_batch{batch_size}_max_iter{max_iters}.data', 'wb') as filehandle:
            pickle.dump([test_losses[best_lr], test_accuracies[best_lr], wt_norms[best_lr], lrs[best_lr]], filehandle)
    else:
        with open(f'data/LrTestRun_BP_modelType{model_type}_data{data}_batch{batch_size}_max_iter{max_iters}.data', 'wb') as filehandle:
            pickle.dump([test_losses, test_accuracies, wt_norms, lrs], filehandle)


