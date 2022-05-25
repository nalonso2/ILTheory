import pickle
import torch
import numpy as np
from utilities import sigmoid_d
from utilities import tanh_d
import matplotlib
import pylab
import math


matplotlib.rcParams['text.usetex']=False
matplotlib.rcParams['savefig.dpi']=400.
matplotlib.rcParams['font.size']=9.0
matplotlib.rcParams['figure.figsize']=(5.0,3.5)
matplotlib.rcParams['axes.formatter.limits']=[-10,10]
matplotlib.rcParams['axes.labelsize']= 9.0
matplotlib.rcParams['figure.subplot.bottom'] = .2
matplotlib.rcParams['figure.subplot.left'] = .2
matplotlib.rcParams["axes.facecolor"] = (0.8, 0.8, 0.8, 0.5)
matplotlib.rcParams['axes.edgecolor'] = 'white'
matplotlib.rcParams['grid.linewidth'] = 1.2
matplotlib.rcParams['grid.color'] = 'white'
matplotlib.rcParams['axes.grid'] = True





############### HELPER FUNCTIONS ##################
def compute_means(data, scale=1):
    with torch.no_grad():
        d_tensor = torch.zeros((len(data), len(data[0]))) + .10

        for m in range(0,len(data)):
            t = torch.tensor(data[m]).view(1,-1).clone()
            d_tensor[m, 0:t.size(1)] = t

        return torch.mean(d_tensor*scale, dim=0)


def compute_nanmeans(data, scale=1):
    with torch.no_grad():

        d_tensor = torch.tensor(data[0]).view(1,-1)
        for m in range(1,len(data)):
            t = torch.tensor(data[m]).view(1,-1).clone()
            '''if t.size(1) == d_tensor.size(1):
                d_tensor = torch.cat((d_tensor, t), dim=0)'''


        return torch.nanmean(d_tensor*scale, dim=0)


def compute_stds(data, scale=1):
    with torch.no_grad():
        d_tensor = torch.zeros((len(data), len(data[0]))) + .10
        for m in range(0, len(data)):
            t = torch.tensor(data[m]).view(1, -1).clone()
            d_tensor[m, 0:t.size(1)] = t

        return torch.std(d_tensor*scale, dim=0)


def compute_means_std_comp(data, scale=1):
    with torch.no_grad():

        d_tensor = torch.tensor(data[0]).view(1,-1)
        for m in range(1,len(data)):
                d_tensor = torch.cat((d_tensor, torch.tensor(data[m]).view(1,-1)), dim=1)

        # return torch.mean(torch.nan_to_num(d_tensor, posinf=pos_inf, nan=nan)).item(), torch.std(torch.nan_to_num(d_tensor, posinf=pos_inf, nan=nan)).item()
        return torch.mean(d_tensor*scale).item(), torch.std(d_tensor*scale).item()


def compute_max_comp(data, scale=1):
    with torch.no_grad():

        d_tensor = torch.tensor(data[0]).view(1,-1)
        for m in range(1,len(data)):
                d_tensor = torch.cat((d_tensor, torch.tensor(data[m]).view(1,-1)), dim=1)

        # return torch.mean(torch.nan_to_num(d_tensor, posinf=pos_inf, nan=nan)).item(), torch.std(torch.nan_to_num(d_tensor, posinf=pos_inf, nan=nan)).item()
        return torch.max(d_tensor*scale).item()


def compute_max_means_std(data, scale=1):
    with torch.no_grad():
        maxs = []
        conv_maxs = []
        for m in range(0,len(data)):
            maxs.append(torch.max(torch.tensor(data[m]) * scale).item())
            conv_maxs.append(torch.argmax(torch.tensor(data[m])).item())

        return [sum(maxs) / len(maxs), torch.std(torch.tensor(maxs)).item(), sum(conv_maxs) / len(conv_maxs)]


def compute_min_means_std(data, scale=1):
    with torch.no_grad():
        maxs = []
        conv_maxs = []
        for m in range(0,len(data)):
            maxs.append(torch.min(torch.tensor(data[m]) * scale).item())
            conv_maxs.append(torch.argmin(torch.tensor(data[m])).item())

        return [sum(maxs) / len(maxs), torch.std(torch.tensor(maxs)).item(), sum(conv_maxs) / len(conv_maxs)]


def compute_layerNorms_means(data):

    with torch.no_grad():
        lNormsMeans = []
        for w in range(6):
            d_tensor = torch.tensor(data[0][w]).view(1,-1)
            for m in range(1,len(data)):
                d_tensor = torch.cat((d_tensor, torch.tensor(data[m][w]).view(1,-1)), dim=0)
            ln = torch.nanmean(d_tensor, dim=0)
            lNormsMeans.append(torch.nanmean(ln).item())

        return lNormsMeans


def compute_layerNorms_stds(data):

    with torch.no_grad():
        lNormsStds = []
        for w in range(len(data[0])):
            d_tensor = torch.tensor(data[0][w]).view(1,-1)
            for m in range(1,len(data)):
                d_tensor = torch.cat((d_tensor, torch.tensor(data[m][w]).view(1,-1)), dim=0)
            ln = torch.nanmean(d_tensor, dim=0)
            lNormsStds.append(torch.std(ln).item())

        return lNormsStds



#Remove Nan values and compute variance
def get_variance(data):
    with torch.no_grad():
        for m in range(len(data)):
            data[m] = torch.tensor(data[m])

        d_tensor = data[0].view(1,-1)
        for m in range(1,len(data)):
            d_tensor = torch.cat((d_tensor, data[m].view(1,-1)), dim=0)

        var = torch.nansum(torch.square(d_tensor - torch.nanmean(d_tensor))) / (torch.numel(d_tensor) - torch.isnan(d_tensor).sum())
        #print(torch.numel(d_tensor), torch.isnan(d_tensor).sum().item(), var.item())

        return var


#Get proportion of nan values in weight norms
def get_nan_prop(data):
    with torch.no_grad():
        for m in range(len(data)):
            data[m] = torch.tensor(data[m])

        d_tensor = data[0].view(1,-1)
        for m in range(1,len(data)):
            d_tensor = torch.cat((d_tensor, data[m].view(1,-1)), dim=0)

        return torch.isnan(d_tensor).sum().item() / torch.numel(d_tensor)


def compute_last_mean(accuracies):

    last_accs = []
    for m in range(len(accuracies)):
        last_accs = last_accs + accuracies[m][-3:]
    mean = sum(last_accs) / len(last_accs)
    std = np.std(np.array(last_accs))

    return 1 - mean, std


def conf_interval(std, n):
    return (std / math.sqrt(n)) * 1.96



def t_test(mean1, mean2, std1, std2, n):

    se1 = std1 / math.sqrt(n)
    se2 = std2 / math.sqrt(n)
    sed = torch.sqrt(torch.square(se1) + torch.square(se2))
    t = (mean1 - mean2) / sed
    return (abs(t) >= 1.96)






####################################################################################################
def plot_SGD_Acc_All_subplots(batch_sz=1, max_iters=[60000, 60000, 50000], nm=False, cont=False):

    with torch.no_grad():
        BPAcc = [[[0,0] for i in range(8)] for x in range(3)]         #[data set][model type]
        BPAccShort = [[[0,0] for i in range(8)] for x in range(3)]

        #Load BP Data
        for d in range(2):
            ms = [0, 1, 2, 6, 7]

            for t in ms:
                with open(f'data/BestLrRun_BP_modelType{t}_data{d}_batch{batch_sz}_max_iter{max_iters[d]}.data', 'rb') as filehandle:
                    BPdata = pickle.load(filehandle)

                BPAcc[d][t][0] = compute_means(BPdata[1], scale=100)
                BPAcc[d][t][1] = compute_stds(BPdata[1], scale=100)


                # This takes 11 evenly space values from the first 2000 training iterations.
                BPAccShort[d][t][0] = [BPAcc[d][t][0][2*x].item() for x in range(21)]
                BPAccShort[d][t][1] = [BPAcc[d][t][1][2*x].item() for x in range(21)]


                # This takes 21 evenly space values from the full list of accuracies so that only 10 are plotted.
                # Otherwise, plot is too noisy/crowded
                freq = int(len(BPAcc[d][t][0]) / 20)
                BPAcc[d][t][0] = [BPAcc[d][t][0][freq * x].item() for x in range(21)]
                BPAcc[d][t][1] = [BPAcc[d][t][1][freq * x].item() for x in range(21)]


        #Plot
        fig, axs = pylab.subplots(2,2, figsize=(8,4))
        data_nms = ['MNIST', 'Fashion-MNIST']


        #BP
        for d in range(2):
            x_axis = torch.linspace(0, max_iters[d], steps=21)
            axs[0, d].set_title(data_nms[d])
            '''if d == 0:
                axs[0, d].set_ylim([93, 98])
            elif d ==1:
                axs[0, d].set_ylim([72, 88])
            else:
                axs[0, d].set_ylim([12, 42])'''

            axs[0, d].errorbar(x_axis, BPAcc[d][0][0], yerr=BPAcc[d][0][1], label='BP-SGD', marker='o', linewidth=1.5, markersize=2.5, alpha=.5, color='black')

            axs[0, d].errorbar(x_axis, BPAcc[d][1][0], yerr=BPAcc[d][1][1], label='BP-Adam', marker='^', linewidth=1.5, markersize=2.5, alpha=.5, color='black', linestyle='--')

            axs[0, d].errorbar(x_axis, BPAcc[d][2][0], yerr=BPAcc[d][2][1], label='IL-SGD', marker='o', linewidth=1.5, markersize=2.5, alpha=.5, color='#1f77b4')

            axs[0, d].errorbar(x_axis, BPAcc[d][6][0], yerr=BPAcc[d][6][1], label='IL-prox', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#ff7f0e')

            axs[0, d].errorbar(x_axis, BPAcc[d][7][0], yerr=BPAcc[d][7][1], label='IL-prox Fast', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#2ca02c')




        # BP Short
        x_axis = torch.linspace(0, 2000, steps=21)
        for d in range(2):
            '''if d == 0:
                axs[1, d].set_ylim([9, 89])
            elif d == 1:
                axs[1, d].set_ylim([9, 65])
            else:
                axs[1, d].set_ylim([9, 23])'''

            axs[1, d].errorbar(x_axis, BPAccShort[d][0][0], yerr=BPAccShort[d][0][1], label='BP-SGD', marker='o', linewidth=1.5, markersize=2.5, alpha=.5, color='black')

            axs[1, d].errorbar(x_axis, BPAccShort[d][1][0], yerr=BPAccShort[d][1][1], label='BP-Adam', marker='^', linewidth=1.5, markersize=2.5, alpha=.5, color='black', linestyle='--')

            axs[1, d].errorbar(x_axis, BPAccShort[d][2][0], yerr=BPAccShort[d][2][1], label='IL-SGD', marker='o', linewidth=1.5, markersize=2.5,
                           alpha=.5, color='#1f77b4')

            axs[1, d].errorbar(x_axis, BPAccShort[d][6][0], yerr=BPAccShort[d][6][1], label='IL-prox', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#ff7f0e')

            axs[1, d].errorbar(x_axis, BPAccShort[d][7][0], yerr=BPAccShort[d][7][1], label='IL-prox Fast', marker='o',
                               linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#2ca02c')






        #pylab.tight_layout()
        axs[0, 0].set(ylabel='Accuracy (%)')
        axs[1, 0].set(ylabel='Accuracy (%)')
        axs[1, 0].set(xlabel='Training Iteration')
        axs[1, 1].set(xlabel='Training Iteration')
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5)
        '''if batch_sz == 1:
            fig.suptitle('Supervised Classification w/ Streaming Data')
        else:
            fig.suptitle('Supervised Classification w/ Mini-batched Data')'''
        #axs[0,0].legend()
        #pylab.suptitle('Online Classification Test Accuracy')
        pylab.savefig(f'plots/BestAccAllSubPlots{max_iters}.png', bbox_inches='tight')
        pylab.close()



##########################################################################
def plot_Acc_Cifar_subplots(batch_szs=[1, 64], max_iters=[50000, 39100], nm=False, cont=False):

    with torch.no_grad():
        BPAcc = [[[0,0] for i in range(10)] for x in range(2)]         #[data type][model type]
        BPAccShort = [[[0,0] for i in range(10)] for x in range(2)]

        #Load BP Data
        for d in range(2):
            ms = [0, 1, 3, 6, 7, 9]

            for t in ms:
                if t != 9 or d == 1:
                    with open(f'data/BestLrRun_BP_modelType{t}_data2_batch{batch_szs[d]}_max_iter{max_iters[d]}.data', 'rb') as filehandle:
                        BPdata = pickle.load(filehandle)


                    BPAcc[d][t][0] = compute_means(BPdata[1], scale=100)
                    BPAcc[d][t][1] = compute_stds(BPdata[1], scale=100)


                    # This takes 11 evenly space values from the first 2000 training iterations.
                    BPAccShort[d][t][0] = [BPAcc[d][t][0][2*x].item() for x in range(21)]
                    BPAccShort[d][t][1] = [BPAcc[d][t][1][2*x].item() for x in range(21)]


                    # This takes 21 evenly space values from the full list of accuracies so that only 10 are plotted.
                    # Otherwise, plot is too noisy/crowded
                    freq = int(len(BPAcc[d][t][0]) / 20)
                    BPAcc[d][t][0] = [BPAcc[d][t][0][freq * x].item() for x in range(21)]
                    BPAcc[d][t][1] = [BPAcc[d][t][1][freq * x].item() for x in range(21)]


        #Plot
        fig, axs = pylab.subplots(2,2, figsize=(10,4))
        data_nms = ['Cifar-10 (Streaming)', 'Cifar-10 (Mini-batched)']


        #streaming
        for d in range(2):
            x_axis = torch.linspace(0, max_iters[d], steps=21)
            axs[0, d].set_title(data_nms[d])
            '''if d == 0:
                axs[0, d].set_ylim([93, 98])
            elif d ==1:
                axs[0, d].set_ylim([72, 88])
            else:
                axs[0, d].set_ylim([12, 42])'''

            axs[0, d].errorbar(x_axis, BPAcc[d][0][0], yerr=BPAcc[d][0][1], label='BP-SGD', marker='o', linewidth=1.5, markersize=2.5, alpha=.5, color='black')

            axs[0, d].errorbar(x_axis, BPAcc[d][1][0], yerr=BPAcc[d][1][1], label='BP-Adam', marker='^', linewidth=1.5, markersize=2.5, alpha=.5, color='black', linestyle='--')

            axs[0, d].errorbar(x_axis, BPAcc[d][3][0], yerr=BPAcc[d][3][1], label='IL-SGD', marker='o', linewidth=1.5, markersize=2.5, alpha=.5, color='#1f77b4')

            if d == 1:
                axs[0, d].errorbar(x_axis, BPAcc[d][9][0], yerr=BPAcc[d][9][1], label='IL-Adam', marker='^', linewidth=1.5, markersize=2.5, alpha=.5, color='#1f77b4', linestyle='--')

            axs[0, d].errorbar(x_axis, BPAcc[d][6][0], yerr=BPAcc[d][6][1], label='IL-prox', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#ff7f0e')

            axs[0, d].errorbar(x_axis, BPAcc[d][7][0], yerr=BPAcc[d][7][1], label='IL-prox Fast', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#2ca02c')



        # mini-batch
        x_axis = torch.linspace(0, 2000, steps=21)
        for d in range(2):
            '''if d == 0:
                axs[1, d].set_ylim([9, 89])
            elif d == 1:
                axs[1, d].set_ylim([9, 65])
            else:
                axs[1, d].set_ylim([9, 23])'''

            axs[1, d].errorbar(x_axis, BPAccShort[d][0][0], yerr=BPAccShort[d][0][1], label='BP-SGD', marker='o', linewidth=1.5, markersize=2.5, alpha=.5, color='black')

            axs[1, d].errorbar(x_axis, BPAccShort[d][1][0], yerr=BPAccShort[d][1][1], label='BP-Adam', marker='^', linewidth=1.5, markersize=2.5, alpha=.5, color='black', linestyle='--')

            axs[1, d].errorbar(x_axis, BPAccShort[d][3][0], yerr=BPAccShort[d][3][1], label='IL-SGD', marker='o', linewidth=1.5, markersize=2.5,
                           alpha=.5, color='#1f77b4')

            if d == 1:
                axs[1, d].errorbar(x_axis, BPAccShort[d][9][0], yerr=BPAccShort[d][9][1], label='IL-Adam', marker='^', linewidth=1.5, markersize=2.5, alpha=.5, color='#1f77b4', linestyle='--')


            axs[1, d].errorbar(x_axis, BPAccShort[d][6][0], yerr=BPAccShort[d][6][1], label='IL-prox', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#ff7f0e')

            axs[1, d].errorbar(x_axis, BPAccShort[d][7][0], yerr=BPAccShort[d][7][1], label='IL-prox Fast', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#2ca02c')






        #pylab.tight_layout()
        axs[0, 0].set(ylabel='Accuracy (%)')
        axs[1, 0].set(ylabel='Accuracy (%)')
        handles, labels = axs[0, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6)
        '''if batch_sz == 1:
            fig.suptitle('Supervised Classification w/ Streaming Data')
        else:
            fig.suptitle('Supervised Classification w/ Mini-batched Data')'''
        #axs[0,0].legend()
        #pylab.suptitle('Online Classification Test Accuracy')
        pylab.savefig(f'plots/BestAccCifarSubplots.png', bbox_inches='tight')
        pylab.close()




#########################################################################################################
def plot_Rec_Loss_subplots(batch_szs=[1, 64],max_iters=[50000, 39100], nm=False, cont=False):

    with torch.no_grad():
        BPAcc = [[[0, 0] for i in range(10)] for x in range(2)]  # [data type][model type]
        BPAccShort = [[[0, 0] for i in range(10)] for x in range(2)]

        #Load BP Data
        for d in range(2):
            ms = [0, 1, 3, 6, 7]
            for t in ms:
                with open(f'data/BestLrRunAE_BP_modelType{t}_data2_batch{batch_szs[d]}_max_iter{max_iters[d]}.data', 'rb') as filehandle:
                    BPdata = pickle.load(filehandle)

                BPAcc[d][t][0] = compute_means(BPdata[0], scale=1)
                BPAcc[d][t][1] = compute_stds(BPdata[0], scale=1)
                print(t, BPdata[-1])



                # This takes 21 evenly space values from the full list of accuracies so that only 10 are plotted.
                # Otherwise, plot is too noisy/crowded
                freq = int(len(BPAcc[d][t][0]) / 20)
                BPAcc[d][t][0] = [BPAcc[d][t][0][freq * x].item() for x in range(21)]
                BPAcc[d][t][1] = [BPAcc[d][t][1][freq * x].item() for x in range(21)]


        #Plot
        fig, axs = pylab.subplots(1,2, figsize=(8,3))


        #BP
        axs[0].set_title('Cifar-10 (Streaming)')
        axs[1].set_title('Cifar-10 (Mini-batched)')
        for d in range(2):
            x_axis = torch.linspace(0, max_iters[d], steps=21)
            '''if d == 0:
                axs[0, d].set_ylim([93, 98])
            elif d ==1:
                axs[0, d].set_ylim([72, 88])
            else:
                axs[0, d].set_ylim([12, 42])'''

            axs[d].errorbar(x_axis, BPAcc[d][0][0], yerr=BPAcc[d][0][1], label='BP-SGD', marker='o', linewidth=1.5, markersize=2.5, alpha=.5, color='black')

            axs[d].errorbar(x_axis, BPAcc[d][1][0], yerr=BPAcc[d][1][1], label='BP-Adam', marker='^', linewidth=1.5, markersize=2.5, alpha=.5, color='black', linestyle='--')

            axs[d].errorbar(x_axis, BPAcc[d][3][0], yerr=BPAcc[d][3][1], label='IL-SGD', marker='o', linewidth=1.5, markersize=2.5, alpha=.5, color='#1f77b4')

            axs[d].errorbar(x_axis, BPAcc[d][6][0], yerr=BPAcc[d][6][1], label='IL-prox', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#ff7f0e')

            axs[d].errorbar(x_axis, BPAcc[d][7][0], yerr=BPAcc[d][7][1], label='IL-prox Fast', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#2ca02c')


        #pylab.tight_layout()
        axs[0].set(ylabel='BCE')
        axs[0].set(xlabel='Iteration')
        axs[1].set(xlabel='Iteration')
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5)
        #axs[0,0].legend()
        #pylab.suptitle('Online Classification Test Accuracy')
        pylab.savefig(f'plots/BestRecLossCifarSubPlots{max_iters}.png', bbox_inches='tight')
        pylab.close()



#####################################################################################################
def plot_Combined_AccLoss_subplots(batch_szs=[1],max_iters=[50000], nm=False, cont=False):
    with torch.no_grad():
        BPAcc = [[[0, 0] for i in range(10)] for x in range(2)]  # [data type][model type]

        #Load BP Data
        ms = [0, 1, 3, 6, 7]
        for t in ms:
            with open(f'data/BestLrRun_BP_modelType{t}_data2_batch1_max_iter50000.data', 'rb') as filehandle:
                BPdata = pickle.load(filehandle)

            BPAcc[0][t][0] = compute_means(BPdata[1], scale=1)
            BPAcc[0][t][1] = compute_stds(BPdata[1], scale=1)

            # This takes 21 evenly space values from the full list of accuracies so that only 21 are plotted.
            # Otherwise, plot is too noisy/crowded
            freq = int(len(BPAcc[0][t][0]) / 20)
            BPAcc[0][t][0] = [BPAcc[0][t][0][freq * x].item() for x in range(21)]
            BPAcc[0][t][1] = [BPAcc[0][t][1][freq * x].item() for x in range(21)]


        for t in ms:
            with open(f'data/BestLrRunAE_BP_modelType{t}_data2_batch1_max_iter50000.data', 'rb') as filehandle:
                BPdata = pickle.load(filehandle)

            BPAcc[1][t][0] = compute_means(BPdata[0], scale=1)
            BPAcc[1][t][1] = compute_stds(BPdata[0], scale=1)

            # This takes 21 evenly spaced values from the full list of accuracies so that only 21 are plotted.
            freq = int(len(BPAcc[1][t][0]) / 20)
            BPAcc[1][t][0] = [BPAcc[1][t][0][freq * x].item() for x in range(21)]
            BPAcc[1][t][1] = [BPAcc[1][t][1][freq * x].item() for x in range(21)]


        #Plot
        fig, axs = pylab.subplots(1,2, figsize=(9.5,2.5))


        #BP
        axs[0].set_title('CIFAR-10 Accuracy')
        axs[1].set_title('CIFAR-10 Reconstruction Loss')
        for d in range(2):
            x_axis = torch.linspace(0, 50000, steps=21)

            axs[d].errorbar(x_axis, BPAcc[d][0][0], yerr=BPAcc[d][0][1], label='BP-SGD', marker='o', linewidth=1.5, markersize=2.5, alpha=.5, color='black')

            axs[d].errorbar(x_axis, BPAcc[d][1][0], yerr=BPAcc[d][1][1], label='BP-Adam', marker='^', linewidth=1.5, markersize=2.5, alpha=.5, color='black', linestyle='--')

            axs[d].errorbar(x_axis, BPAcc[d][3][0], yerr=BPAcc[d][3][1], label='IL-SGD', marker='o', linewidth=1.5, markersize=2.5, alpha=.5, color='#1f77b4')

            axs[d].errorbar(x_axis, BPAcc[d][6][0], yerr=BPAcc[d][6][1], label='IL-prox', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#ff7f0e')

            axs[d].errorbar(x_axis, BPAcc[d][7][0], yerr=BPAcc[d][7][1], label='IL-prox Fast', marker='o', linewidth=1.5, markersize=2.5,
                               alpha=.5, color='#2ca02c')


        #pylab.tight_layout()
        axs[0].set(ylabel='Test Accuracy (%)')
        axs[1].set(ylabel='Test BCE')
        axs[0].set(xlabel='Iteration')
        axs[1].set(xlabel='Iteration')
        '''handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', ncol=2)'''
        axs[0].legend(ncol=2)
        #pylab.suptitle('Online Classification Test Accuracy')
        pylab.savefig(f'plots/BestCombCifarSubPlots{max_iters}.png', bbox_inches='tight')
        pylab.close()


#####################################################################################################
def plot_SGD_LR_Stab_subplots(alpha, batch_sz=1, smax=True):

    with torch.no_grad():

        BPAcc = [[[] for i in range(5)] for x in range(3)]         #[data set][model type][avg, std]
        DTPAcc = [[[] for i in range(5)] for x in range(3)]
        BPAccSTD = [[[] for i in range(5)] for x in range(3)]      #[data set][model type][avg, std]
        DTPAccSTD = [[[] for i in range(5)] for x in range(3)]

        #Load BP Data
        for d in range(3):
            for lr in alpha[0]:
                with open(f'data/SGDTests_BP_lr{lr}_data{d}_batch{batch_sz}_smax{smax}.data', 'rb') as filehandle:
                    BPdata = pickle.load(filehandle)

                # Load Accuracies for each model kind
                for i in range(5):
                    BPAcc[d][i].append(compute_max_means_std(BPdata[1][i], scale=100)[0])
                    BPAccSTD[d][i].append(compute_max_means_std(BPdata[1][i], scale=100)[1])


        #Load DTP Data
        for d in range(3):
            for lr in alpha[1]:
                with open(f'data/SGDTests_DTP_lr{lr}_data{d}_batch{batch_sz}_smax{smax}.data', 'rb') as filehandle:
                    DTPdata = pickle.load(filehandle)

                    # Load Accuracies
                    for i in range(5):
                        DTPAcc[d][i].append(compute_max_means_std(DTPdata[1][i], scale=100)[0])
                        DTPAccSTD[d][i].append(compute_max_means_std(DTPdata[1][i], scale=100)[1])



        #Plot
        fig, axs = pylab.subplots(2,3, figsize=(11,5.5))
        colors = [['black', 'blue', 'blue', 'red', 'red'], ['black', 'green', 'green', 'orange', 'orange']]
        names = [['BP-SGD', 'G-IL SGD1', 'G-IL SGD2', 'G-IL SGD1 (cont.)', 'G-IL SGD2 (cont.)'], ['DTP', 'G-IL DTP1', 'G-IL DTP2', 'G-IL DTP1 (cont.)', 'G-IL DTP2 (cont.)']]
        data_nms = ['MNIST', 'Fashion-MNIST', 'Cifar-10']

        #BP LRs
        for d in range(3):
            axs[0, d].set_title(data_nms[d])
            axs[0, d].set_ylim([-.05, 105])
            for m in range(5):


                if m == 2 or m == 4:
                    axs[0, d].plot(alpha[0], BPAcc[d][m], label=names[0][m], marker='o', linewidth=2.25,
                                   markersize=2, alpha=.5, color=colors[0][m], ls='--')
                else:
                    axs[0, d].plot(alpha[0], BPAcc[d][m], label=names[0][m], marker='^', linewidth=2.25,
                                       markersize=2, alpha=.5, color=colors[0][m])

                axs[0, d].errorbar(alpha[0], BPAcc[d][m], yerr=BPAccSTD[d][m], alpha=.5,
                                   color=colors[0][m], ls='None')


        axs[0, 0].set(ylabel='Accuracy (%)')
        axs[0, 2].legend(ncol=1)

        # DTP LRs
        for d in range(3):
            axs[1, d].set_ylim([-.05, 105])
            axs[1, d].set(xlabel='Learning Rate')
            for m in range(5):
                if m == 2 or m == 4:
                    axs[1, d].plot(alpha[1], DTPAcc[d][m], label=names[1][m], marker='o', linewidth=2.25,
                                   markersize=2, alpha=.5, color=colors[1][m], ls='--')
                else:
                    axs[1, d].plot(alpha[1], DTPAcc[d][m], label=names[1][m], marker='o', linewidth=2.25,
                                   markersize=2, alpha=.5, color=colors[1][m])

                axs[1, d].errorbar(alpha[1], DTPAcc[d][m], yerr=DTPAccSTD[d][m], alpha=.5,
                                   color=colors[1][m], ls='None')

        axs[1, 0].set(ylabel='Accuracy (%)')
        axs[1, 2].legend(ncol=1)



        #pylab.tight_layout()
        pylab.savefig(f'plots/lrStabil.png', bbox_inches='tight')
        pylab.close()



##############################################################################################

def plot_toy_out_acc():
    with open(f'data/Toy_BP_PC_LinearTrue.data', 'rb') as filehandle:
        dataBP = pickle.load(filehandle)

    with open(f'data/Toy_BP_PC_LinearFalse.data', 'rb') as filehandle:
        dataBPrelu = pickle.load(filehandle)

    with open(f'data/Toy_DTP_PC.data', 'rb') as filehandle:
        dataDTP = pickle.load(filehandle)



    fig, axs = pylab.subplots(1, 3, figsize=(6, 2))

    axs[1].set(xlabel='Learning Rate')

    axs[0].plot(dataBP[-1], np.array(dataBP[0])[:, 0], label='BP-SGD', marker='o', linewidth=2.25, markersize=2, alpha=.5, color='black')
    axs[0].errorbar(dataBP[-1], np.array(dataBP[0])[:, 0], yerr=conf_interval(np.array(dataBP[0])[:, 1], n=10000), alpha=.5, color='black')
    axs[0].plot(dataBP[-1], np.array(dataBP[1])[:, 0], label='IL-SGD', marker='o', linewidth=2.25, markersize=2, alpha=.5, color='r')
    axs[0].errorbar(dataBP[-1], np.array(dataBP[1])[:, 0], yerr=conf_interval(np.array(dataBP[1])[:, 1], n=10000), alpha=.5, color='r')

    axs[1].plot(dataBPrelu[-1], np.array(dataBPrelu[0])[:, 0], label='BP-SGD', marker='o', linewidth=2.25, markersize=2,alpha=.5, color='black')
    axs[1].errorbar(dataBPrelu[-1], np.array(dataBPrelu[0])[:, 0], yerr=conf_interval(np.array(dataBPrelu[1])[:, 1], n=10000), alpha=.5, color='black')
    axs[1].plot(dataBPrelu[-1], np.array(dataBPrelu[1])[:, 0], label='IL-SGD', marker='o', linewidth=2.25, markersize=2, alpha=.5, color='r')
    axs[1].errorbar(dataBPrelu[-1], np.array(dataBPrelu[1])[:, 0], yerr=conf_interval(np.array(dataBPrelu[1])[:, 1], n=10000), alpha=.5, color='r')

    axs[2].plot(dataDTP[-1], np.array(dataDTP[0])[:, 0], label='BP-GN', marker='o', linewidth=2.25, markersize=3, alpha=.5, color='black')
    axs[2].errorbar(dataDTP[-1], np.array(dataDTP[0])[:, 0], yerr=conf_interval(np.array(dataDTP[1])[:, 1], n=10000), alpha=.5, color='black')
    axs[2].plot(dataDTP[-1], np.array(dataDTP[1])[:, 0], label='IL-GN', marker='o', linewidth=2.25, markersize=2, alpha=.5, color='g')
    axs[2].errorbar(dataDTP[-1], np.array(dataDTP[1])[:, 0], yerr=conf_interval(np.array(dataDTP[1])[:, 1], n=10000), alpha=.5, color='g')


    '''print('T-TESTS')
    print(t_test(torch.tensor(dataBP[0])[:, 0], torch.tensor(data[1])[:, 0], torch.tensor(data[0])[:, 1],torch.tensor(data[1])[:, 1], n=10000))
    print(t_test(torch.tensor(dataBP[2])[:, 0], torch.tensor(data[3])[:, 0], torch.tensor(data[2])[:, 1], torch.tensor(data[3])[:, 1], n=10000))
    print(t_test(torch.tensor(dataBP[4])[:, 0], torch.tensor(data[5])[:, 0], torch.tensor(data[4])[:, 1], torch.tensor(data[5])[:, 1], n=10000))'''


    axs[0].set(ylabel='$\dfrac{\Delta h_N \cdot \Delta h_N^{Min}} {\Vert \Delta h_N \Vert \Vert \Delta h_N^{Min} \Vert}$')
    axs[0].set_xlim([-0.08, 5.1])
    axs[0].set_ylim([.46, 1.05])
    axs[1].set_xlim([-0.08, 5.1])
    axs[1].set_ylim([.46, 1.05])
    axs[2].set_xlim([-0.02, .8])
    axs[2].set_ylim([.46, 1.05])
    axs[0].title.set_text('Linear SGD')
    axs[1].title.set_text('ReLU SGD')
    axs[2].title.set_text('Linear GN')
    axs[0].legend(loc='best', ncol=1)
    #axs[1].legend(loc='best', ncol=1, fancybox=True)
    axs[2].legend(loc='lower right', ncol=1)
    pylab.tight_layout()
    pylab.savefig(f'plots/toy_Out_Ang.png', bbox_inches='tight')
    pylab.close()



##################################################################################################################

def plot_toy_delta_out():
    with open(f'data/Toy_BP_PC_LinearTrue.data', 'rb') as filehandle:
        dataBP = pickle.load(filehandle)

    with open(f'data/Toy_BP_PC_LinearFalse.data', 'rb') as filehandle:
        dataBPrelu = pickle.load(filehandle)


    fig, axs = pylab.subplots(1, 2, figsize=(5, 2))

    axs[0].set(xlabel='Learning Rate')
    axs[1].set(xlabel='Learning Rate')

    tens_data_bp = dataBP[2][0].view(4,1).clone()
    tens_data_il = dataBP[3][0].view(4, 1).clone()
    for x in range(1, len(dataBP[2])):
        tens_data_bp = torch.cat((tens_data_bp, dataBP[2][x].view(4,1).clone()), dim=1)
        tens_data_il = torch.cat((tens_data_il, dataBP[3][x].view(4, 1).clone()), dim=1)

    axs[0].plot(dataBP[-1], torch.mean(tens_data_bp, dim=0), label='BP-SGD', marker='o', linewidth=2.25, markersize=2, alpha=.7, color='black')
    axs[0].plot(dataBP[-1], torch.mean(tens_data_il, dim=0), label='IL-SGD', marker='o', linewidth=2.25, markersize=2, alpha=.7, color='r')

    tens_data_bprlu = dataBP[2][0].view(4, 1).clone()
    tens_data_ilrlu = dataBP[3][0].view(4, 1).clone()
    for x in range(1, len(dataBP[2])):
        tens_data_bprlu = torch.cat((tens_data_bprlu, dataBPrelu[2][x].view(4, 1).clone()), dim=1)
        tens_data_ilrlu = torch.cat((tens_data_ilrlu, dataBPrelu[3][x].view(4, 1).clone()), dim=1)

    axs[1].plot(dataBPrelu[-1], torch.mean(tens_data_bprlu, dim=0), label='BP-SGD', marker='o', linewidth=2, markersize=2,
                alpha=.7, color='black')
    axs[1].plot(dataBPrelu[-1], torch.mean(tens_data_ilrlu, dim=0), label='IL-SGD', marker='o', linewidth=2, markersize=2,
                alpha=.7, color='r')


    axs[0].set(ylabel='$\Delta W$ Compatibility Score')
    axs[0].set_xlim([-0.01, .41])
    axs[0].set_ylim([-.01, .55])
    axs[1].set_xlim([-0.01, .41])
    axs[1].set_ylim([-.01, .55])
    axs[0].title.set_text('Linear SGD')
    axs[1].title.set_text('ReLU SGD')
    axs[0].legend(loc='best', ncol=1)
    #axs[1].legend(loc='best', ncol=1, fancybox=True)
    pylab.tight_layout()
    pylab.savefig(f'plots/toy_Delta_Out.png', bbox_inches='tight')
    pylab.close()





##################################################################################################################
def compute_best_acc_table(max_iters=60000, data=0, batch_sz=1):

    for t in [0,1,3,6,7,9,10]:
        print(f'\nBP TYPE{t}')
        with open(f'data/BestLrRun_BP_modelType{t}_data{data}_batch{batch_sz}_max_iter{max_iters}.data', 'rb') as filehandle:
            BPdata = pickle.load(filehandle)

        '''acc_avg = torch.mean(torch.tensor([BPdata[1][x][-1]*100 for x in range(len(BPdata[1]))]))
        acc_std = torch.std(torch.tensor([BPdata[1][x][-1]*100 for x in range(len(BPdata[1]))]))'''
        pr_data = compute_max_means_std(BPdata[1], scale=100)
        acc_avg = pr_data[0]
        acc_std = pr_data[1]

        print(f'Acc:{round(acc_avg, 3)} ({round(acc_std, 3)})')
        '''if t == 0:
            bpsgd_avg = acc_avg.clone()
            bpsgd_std = acc_std.clone()'''

        wtnm_avg = compute_means(BPdata[2])
        wtnm_std = compute_stds(BPdata[2])
        wtnm_max = compute_max_comp(BPdata[2])
        print(f'Weight Norm:{round(wtnm_avg[-1].item(), 4)} ({round(wtnm_std[-1].item(), 4)}) Max: {wtnm_max}')

        conv = torch.argmax(compute_means(BPdata[1])) * 50
        print('Converge Iteration:', conv)

        '''t = t_test(bpsgd_avg, acc_avg, bpsgd_std, acc_std, n=len(BPdata[1]))
        print('T-Test:', t)'''

        print('LR:', BPdata[-1])


##################################################################################################################
def compute_best_accConv_table(max_iters=60000, batch_sz=1, models=[1,3,4]):

    for t in models:
        print(f'\nBP TYPE{t}')
        with open(f'data/BestLrRun_BPConv_modelType{t}_batch{batch_sz}_max_iter{max_iters}.data', 'rb') as filehandle:
            BPdata = pickle.load(filehandle)

        '''acc_avg = torch.mean(torch.tensor([BPdata[1][x][-1]*100 for x in range(len(BPdata[1]))]))
        acc_std = torch.std(torch.tensor([BPdata[1][x][-1]*100 for x in range(len(BPdata[1]))]))'''
        acc_avg = compute_max_means_std(BPdata[1], scale=100)
        #print(f'Acc:{round(acc_avg.item(), 3)} ({round(acc_std.item(), 3)})')
        print(f'Acc:{round(acc_avg[0], 3)} ({round(acc_avg[1], 3)})')
        '''if t == 0:
            bpsgd_avg = acc_avg.clone()
            bpsgd_std = acc_std.clone()'''

        wtnm_avg = compute_means(BPdata[2])
        wtnm_std = compute_stds(BPdata[2])
        wtnm_max = compute_max_comp(BPdata[2])
        print(f'Weight Norm:{round(wtnm_avg[-1].item(), 4)} ({round(wtnm_std[-1].item(), 4)}) Max: {wtnm_max}')

        conv = torch.argmax(compute_means(BPdata[1])) * 50
        print('Conv Iteration:', conv)

        '''t = t_test(bpsgd_avg, acc_avg, bpsgd_std, acc_std, n=len(BPdata[1]))
        print('T-Test:', t)'''

        print('LR:', BPdata[-1])



##################################################################################################################
def compute_best_rec_table(max_iters=60000, data=0, batch_sz=1):

    for t in [0,1,3,6,7]:
        print(f'\nBP TYPE{t}')
        with open(f'data/BestLrRunAE_BP_modelType{t}_data{data}_batch{batch_sz}_max_iter{max_iters}.data', 'rb') as filehandle:
            BPdata = pickle.load(filehandle)

        '''acc_avg = torch.mean(torch.tensor([BPdata[0][x][-1] for x in range(len(BPdata[0]))]))
        acc_std = torch.std(torch.tensor([BPdata[0][x][-1] for x in range(len(BPdata[0]))]))'''
        pr_data = compute_min_means_std(BPdata[0], scale=1)
        acc_avg = pr_data[0]
        acc_std = pr_data[1]
        print(f'Loss:{round(acc_avg, 3)} ({round(acc_std, 3)})')
        '''if t == 0:
            bpsgd_avg = acc_avg.clone()
            bpsgd_std = acc_std.clone()'''

        wtnm_avg = compute_means(BPdata[1])
        wtnm_std = compute_stds(BPdata[1])
        wtnm_max = compute_max_comp(BPdata[1])
        print(f'Weight Norm:{round(wtnm_avg[-1].item(), 4)} ({round(wtnm_std[-1].item(), 4)}) Max: {wtnm_max}')

        conv = torch.argmax(compute_means(BPdata[0])) * 50
        print('Converge Iteration:', conv)

        '''t = t_test(bpsgd_avg, acc_avg, bpsgd_std, acc_std, n=len(BPdata[1]))
        print('T-Test:', t)'''

        print('LR:', BPdata[-1])



##################################################################################################################
def compute_stability_table_sgd(max_iters=15000, data=0, batch_sz=1):

    for t in [0, 3, 6, 8]:
        print(f'\nBP TYPE{t}')
        with open(f'data/LrTestRun_BP_modelType{t}_data{data}_batch{batch_sz}_max_iter{max_iters}.data', 'rb') as filehandle:
            BPdata = pickle.load(filehandle)
        for lr in reversed(range(len(BPdata[-1]))):
            #acc_avg = compute_means(BPdata[1][lr], scale=100)
            acc_avg = torch.mean(torch.tensor([BPdata[1][lr][x][-1]*100 for x in range(len(BPdata[1][lr]))]))
            #acc_std = compute_stds(BPdata[1][lr], scale=100)
            acc_std = torch.std(torch.tensor([BPdata[1][lr][x][-1]*100 for x in range(len(BPdata[1][lr]))]))
            print(f'Acc(lr={BPdata[-1][lr]}):{round(acc_avg.item(), 3)} ({round(acc_std.item(), 3)})')

            wtnm_avg = compute_means(BPdata[2][lr])
            wtnm_std = compute_stds(BPdata[2][lr])
            wtnm_max = compute_max_comp(BPdata[2][lr])
            print(f'Weight Norm (lr={BPdata[-1][lr]}):{round(wtnm_avg[-1].item(), 4)} ({round(wtnm_std[-1].item(), 4)}) Max: {wtnm_max}')


##################################################################################################
def plot_stabil_wt_nm(max_iters=15000, data=0):

    wt_nm_avg = [[[] for m in range(4)] for x in range(2)]
    wt_nm_std = [[[] for m in range(4)] for x in range(2)]
    bn = 0
    for b in [64,1]:
        m = 0
        for t in [0, 2, 3, 6]:
            print(f'\nBP TYPE{t}')
            with open(f'data/LrTestRun_BP_modelType{t}_data{data}_batch{b}_max_iter{max_iters}.data', 'rb') as filehandle:
                BPdata = pickle.load(filehandle)

            for lr in reversed(range(len(BPdata[-1]))):
                if BPdata[-1][lr] != 5:
                    print(BPdata[-1][lr])
                    wt_avg, wt_std = compute_means_std_comp(BPdata[2][lr])
                    wt_nm_avg[bn][m].append(wt_avg)
                    wt_nm_std[bn][m].append(wt_std**2)

            print(wt_nm_avg[bn][m])
            m += 1
        bn +=1



    fig, axs = pylab.subplots(2, 2, figsize=(6, 3))
    x = [.01, .1, 1, 2.5, 10, 100]

    for ax in range(2):
        axs[0,ax].plot(x, wt_nm_avg[ax][1], label='IL-SGD1', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[0,ax].plot(x, wt_nm_avg[ax][2], label='IL-SGD2', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[0,ax].plot(x, wt_nm_avg[ax][3], label='IL-SGD prox', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[0,ax].plot(x, wt_nm_avg[ax][0], label='BP-SGD', marker='o', linewidth=2.25, markersize=3, alpha=.5, color='black')

    for ax in range(2):
        axs[1,ax].plot(x, wt_nm_std[ax][1], label='IL-SGD1', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[1,ax].plot(x, wt_nm_std[ax][2], label='IL-SGD2', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[1,ax].plot(x, wt_nm_std[ax][3], label='IL-SGD prox', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[1,ax].plot(x, wt_nm_std[ax][0], label='BP-SGD', marker='o', linewidth=2.25, markersize=3, alpha=.5, color='black')


    axs[1,0].set(xlabel='Learning Rate')
    axs[0,0].set_xscale('log')
    axs[1, 0].set_xscale('log')
    axs[0, 1].set_xscale('log')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set(xlabel='Learning Rate')
    axs[0,0].set(ylabel=r'$\Vert \Delta \theta \Vert$')
    axs[1, 0].set(ylabel=r'$Var(\Vert \Delta \theta \Vert)$')

    #axs[0].set_xlim([0.00001, 150])
    axs[0,0].set_ylim([0, .09])
    axs[0,1].set_ylim([0, .09])
    axs[1, 0].set_ylim([-0.0005, .009])
    axs[1, 1].set_ylim([-0.0005, .009])
    #axs[1].set_xlim([0.00001, 150])

    axs[0,0].set_title('Mini-Batch Training')
    axs[0,1].set_title('Online Training')

    handles, labels = axs[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)


    pylab.tight_layout()
    pylab.savefig(f'plots/wtnms.png', bbox_inches='tight')
    pylab.close()



############################################################################################################



def plot_activity_sims(data=0):
    with torch.no_grad():

        h_sims = []
        h_mses = []
        prox = []

        for t in range(3):
            print(f'\nBP TYPE{t}')
            with open(f'data/wtUpAnalysis_BP_modelType{t}_data{data}.data', 'rb') as filehandle:
                wt_data = pickle.load(filehandle)

                h_sims.append(torch.nanmean(torch.nanmean(torch.tensor(wt_data[2]), dim=0), dim=0))
                h_mses.append(torch.nanmean(torch.nanmean(torch.tensor(wt_data[3]).sqrt(), dim=0), dim=0))
                prox.append(torch.nanmean(torch.nanmean(torch.tensor(wt_data[4]), dim=0), dim=0))






        fig, axs = pylab.subplots(1, 3, figsize=(7, 2))
        pylab.setp(axs, xticks=[0,1], xticklabels=['$\hat{h}^{(b)} (Initial)$', '$\hat{h}^{(b)} (Final)$'])


        x = [0,1]

        axs[0].plot(x, h_sims[0], label='IL-SGD1', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[1].plot(x, h_mses[0], label='IL-SGD1', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[2].plot(x, prox[0], label='IL-SGD1', marker='o', linewidth=2.25, markersize=3, alpha=.5)

        axs[0].plot(x, h_sims[1], label='IL-SGD2', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[1].plot(x, h_mses[1], label='IL-SGD2', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[2].plot(x, prox[1], label='IL-SGD2', marker='o', linewidth=2.25, markersize=3, alpha=.5)

        axs[0].plot(x, h_sims[2], label='IL-SGD prox', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[1].plot(x, h_mses[2], label='IL-SGD prox', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[2].plot(x, prox[2], label='IL-SGD prox', marker='o', linewidth=2.25, markersize=3, alpha=.5)


        #axs[0].errorbar(dataBP[-1], np.array(dataBP[0])[:, 0], yerr=conf_interval(np.array(dataBP[5])[:, 1], n=10000), alpha=.5)
        #axs[1].errorbar(dataDTP[-1], np.array(dataDTP[0])[:, 0], yerr=conf_interval(np.array(dataDTP[5])[:, 1], n=10000), alpha=.5, color='black')



        axs[0].set(ylabel='cos($\hat{h}^{(b)}, h^{(b+1)}$)')
        axs[1].set(ylabel='Eucl-Dis($\hat{h}^{(b)}, h^{(b+1)}$)')
        axs[2].set(ylabel=r'$L(\theta) + \frac{1}{2 \alpha} \Vert \Delta \theta \Vert$')
        #axs[0].set_ylim([.929, .975])
        #axs[1].set_ylim([1.3, 3.05])
        axs[2].set_ylim([-0.01, .75])
        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
        pylab.tight_layout()
        pylab.savefig(f'plots/h_sims.png', bbox_inches='tight')
        pylab.close()



######################################################################################################
def plot_prox(data=0):
    with torch.no_grad():
        proxFF = []
        proxFull = []
        proxRnd = []
        proxRndFull = []

        for t in range(3):
            print(f'\nBP TYPE{t}')
            with open(f'data/wtUpAnalysis2_BP_modelType{t}_data{data}.data', 'rb') as filehandle:
                wt_data = pickle.load(filehandle)

                proxFF.append(torch.nanmean(torch.nanmean(torch.tensor(wt_data[0]), dim=0), dim=0))
                proxFull.append(torch.nanmean(torch.nanmean(torch.tensor(wt_data[1]), dim=0), dim=0))
                proxRnd.append(torch.nanmean(torch.nanmean(torch.tensor(wt_data[2]), dim=0), dim=0))
                proxRndFull.append(torch.nanmean(torch.nanmean(torch.tensor(wt_data[3]), dim=0), dim=0))






        fig, axs = pylab.subplots(1, 4, figsize=(8, 2))
        pylab.setp(axs, xticks=[0,1], xticklabels=['$\hat{h}(Initial)$', '$\hat{h}(Final)$'])


        x = [0,1]

        axs[0].plot(x, proxFF[0], label='IL-SGD1', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[1].plot(x, proxFull[0], label='IL-SGD1', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[2].plot(x, proxRnd[0], label='IL-SGD1', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[3].plot(x, proxRndFull[0], label='IL-SGD1', marker='o', linewidth=2.25, markersize=3, alpha=.5)

        axs[0].plot(x, proxFF[1], label='IL-SGD2', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[1].plot(x, proxFull[1], label='IL-SGD2', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[2].plot(x, proxRnd[1], label='IL-SGD2', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[3].plot(x, proxRndFull[1], label='IL-SGD2', marker='o', linewidth=2.25, markersize=3, alpha=.5)

        axs[0].plot(x, proxFF[2], label='IL-SGD prox', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[1].plot(x, proxFull[2], label='IL-SGD prox', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[2].plot(x, proxRnd[2], label='IL-SGD prox', marker='o', linewidth=2.25, markersize=3, alpha=.5)
        axs[3].plot(x, proxRndFull[2], label='IL-SGD prox', marker='o', linewidth=2.25, markersize=3, alpha=.5)



        axs[0].set(ylabel=r'$L(\theta) + \frac{1}{2 \alpha} \Vert \Delta \theta \Vert$')
        axs[0].set_ylim(-0.1, 2.6)
        axs[1].set_ylim(-0.1, 2.6)
        axs[0].set_title('FF Init.')
        axs[1].set_title('FF + Full Clamp')
        axs[2].set_title('Rand. Init.')
        axs[3].set_title('Rand. + Full Clamp')
        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
        pylab.tight_layout()
        pylab.savefig(f'plots/prox_init.png', bbox_inches='tight')
        pylab.close()





######################################################################################################
def plot_wt_sims(data=0):
    with torch.no_grad():
        bp_wt_sims = []
        ibp_wt_sims  = []

        for t in range(3):
            with open(f'data/wtUpAnalysis_BP_modelType{t}_data{data}.data', 'rb') as filehandle:
                wt_data = pickle.load(filehandle)

                bp_wt_sims.append(1 - compute_nanmeans(wt_data[-2]))
                ibp_wt_sims.append(1 - compute_nanmeans(wt_data[-1]))


        fig, axs = pylab.subplots(1, 2, figsize=(5, 2))

        x = [0,1]

        axs[0].plot(bp_wt_sims[0], label='IL-SGD1', linewidth=1.25, alpha=.8)
        axs[1].plot(ibp_wt_sims[0], label='IL-SGD1', linewidth=1.25, alpha=.8)

        axs[0].plot(bp_wt_sims[1], label='IL-SGD2', linewidth=1.25, alpha=.5)
        axs[1].plot(ibp_wt_sims[1], label='IL-SGD2', linewidth=1.25, alpha=.5)

        axs[0].plot(bp_wt_sims[2], label='IL-SGD prox', linewidth=1.25, alpha=.5)
        axs[1].plot(ibp_wt_sims[2], label='IL-SGD prox', linewidth=1.25, alpha=.5)


        axs[0].set(ylabel=r'$\Delta W_{5}^{IL} \Delta W_{5}^{BP} $')
        axs[1].set(ylabel=r'$\Delta W_{5}^{IL} \Delta W_{5}^{impBP} $')
        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
        pylab.tight_layout()
        pylab.savefig(f'plots/wt_sims.png', bbox_inches='tight')
        pylab.close()


#plot_SGD_Acc_All_subplots()
#plot_toy_out_acc()
#plot_toy_delta_out()
#compute_best_acc_table(data=2, max_iters=50000, batch_sz=1)
#compute_best_accConv_table(max_iters=77000, batch_sz=64, models=[1,3,4])
compute_best_rec_table(data=2, max_iters=50000, batch_sz=1)
#compute_stability_table_sgd(data=0, max_iters=20000, batch_sz=1)
#plot_Rec_Loss_subplots()
#plot_Acc_Cifar_subplots()
#plot_Combined_AccLoss_subplots()