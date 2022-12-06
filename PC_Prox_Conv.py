import torch
from torch import nn
from utilities import sigmoid_d
from utilities import tanh_d

relu = torch.nn.ReLU()
l_relu = nn.LeakyReLU(0.5)
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
NLL = nn.NLLLoss(reduction='sum')
softmax = torch.nn.Softmax(dim=1)

class PC_prox(nn.Module):

    def __init__(self, lr=100, func=relu, t_type=0, n_iter=25, bot_gamma=.07, top_gamma=.07,
                 rand_init=False, crossEnt=True, adam=False, smax=True, adam_lr=.0001, eps=0, bias=True):
        super().__init__()

        self.num_layers = 5
        self.n_iter = n_iter
        self.l_rate = lr
        self.target_type = t_type  # 0=BP, 1=OD-BP, 2=PC
        self.func = func
        self.bot_gamma = bot_gamma
        self.top_gamma = top_gamma
        self.rand_init = rand_init
        self.crossEnt = crossEnt
        self.adam = adam
        self.adam_lr = adam_lr
        self.eps = eps
        self.smax = smax
        self.bias = bias

        self.wts = nn.Sequential(

            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), bias=True),
            ),

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), bias=True),
            ),

            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=True),
                nn.Flatten()
            ),

            nn.Sequential(
                nn.ReLU(),
                nn.Linear(1024, 10, bias=True)
            )
        )

        self.optims = self.make_optims()
        self.bp_optim = torch.optim.Adam(self.wts.parameters(), lr=self.adam_lr)

    def make_optims(self):

        f = []
        for n in range(len(self.wts)):
            if self.adam:
                f.append(torch.optim.Adam(self.wts[n].parameters(), lr=self.adam_lr))
            else:
                f.append(torch.optim.SGD(self.wts[n].parameters(), lr=1))
        return f

    ############################## COMPUTE FORWARD VALUES ##################################
    def compute_values(self, x):
        with torch.no_grad():
            h = [torch.randn(1, 1) for i in range(self.num_layers)]

            # First h is the input
            h[0] = x.clone()

            # Compute all hs except last one. Use Tanh as in paper
            for i in range(1, self.num_layers):
                h[i] = self.wts[i - 1](h[i - 1].detach())
        return h

    ############################## PC TARGETS ##################################
    def compute_PC_targets(self, h, global_target, y=None):
        with torch.no_grad():
            targ = [h[i].clone() for i in range(self.num_layers)]
            eps = [torch.zeros_like(h[i]) for i in range(self.num_layers)]
            p = [h[i].clone() for i in range(self.num_layers)]
            nm = torch.mean(torch.square(relu(h[-2])).sum(1)).item() + self.eps + self.bias
            targ[-1] = (1 / (1 + nm * self.l_rate)) * softmax(h[-1]) + (nm * self.l_rate / (1 + nm * self.l_rate)) * global_target.clone()

        # Iterative updates
        for i in range(self.n_iter):
            self.bp_optim.zero_grad()

            # Compute errors
            with torch.no_grad():
                for layer in range(1, self.num_layers - 1):
                    eps[layer] = targ[layer] - p[layer]  # MSE gradient
                if self.smax:
                    eps[-1] = (targ[-1] - softmax(p[-1])) * .5  # Cross-ent w/ softmax gradient
                else:
                    eps[-1] = (targ[-1] - torch.sigmoid(p[-1]))  # Binary cross-ent with sigmoid

                '''print(i, round(torch.square(eps[-1]).sum().item(), 5),
                      round(torch.square(eps[-2]).sum().item(), 5),
                      round(torch.square(eps[-3]).sum().item(), 5),
                      round(torch.square(eps[-4]).sum().item(), 5),
                      round(sum([torch.square(eps[l]).sum().item() for l in range(len(eps))]), 5))'''

            # Update Targets
            for layer in range(1, self.num_layers - 1):
                _, epsdfdt = torch.autograd.functional.vjp(self.wts[layer], targ[layer], eps[layer + 1])
                with torch.no_grad():
                    dt = self.top_gamma * -eps[layer] + self.bot_gamma * epsdfdt
                    targ[layer] = targ[layer] + dt

            with torch.no_grad():
                nm = torch.mean(torch.square(relu(targ[-2])).sum()).item() + self.bias + self.eps
                targ[-1] = (1 / (1 + nm*self.l_rate)) * softmax(p[-1]) + (nm*self.l_rate / (1 + nm*self.l_rate)) * global_target.clone()

            # Compute new Predictions
            with torch.no_grad():
                for layer in range(1, self.num_layers):
                    p[layer] = self.wts[layer - 1](targ[layer - 1].detach())

        return targ

    ############################## GENERAL TARGET PROP FUNCTION ##################################
    def compute_targets(self, h, global_target, y=None):
        if self.target_type == 1:
            targets = self.compute_OD_BP_targets(h, global_target)

        elif self.target_type == 2:
            targets = self.compute_PC_targets(h, global_target, y=y)

        else:
            targets = None

        return targets

    ############################## TRAIN ##################################
    def train_wts(self, x, global_target, y, get_grads=False):

        # Record params before update
        with torch.no_grad():
            old_wts = [self.wts[0][0].weight.data.clone()]
            old_params = self.wts[0][0].weight.data.clone().view(-1)
            for i in range(1, self.num_layers - 1):
                old_wts.append(self.wts[i][1].weight.data.clone())
                old_params = torch.cat((old_params, self.wts[i][1].weight.data.clone().view(-1)), dim=0)

        # Get feedforward and target values
        h = self.compute_values(x)
        h_hat = self.compute_targets(h, global_target, y)

        # Update weights
        if (self.target_type == 1 or self.target_type == 2):
            self.PC_update(h_hat, y)
        elif self.target_type == 0:
            self.BP_update(x, y, global_target)

        # Record new params
        with torch.no_grad():
            new_wts = [self.wts[0][0].weight.data.clone()]
            new_params = self.wts[0][0].weight.data.clone().view(-1)
            for i in range(1, self.num_layers - 1):
                new_wts.append(self.wts[i][1].weight.data.clone())
                new_params = torch.cat((new_params, self.wts[i][1].weight.data.clone().view(-1)), dim=0)

        # return h, change in individual weights, and change in param vector
        return h, h_hat, [new_wts[w] - old_wts[w] for w in range(self.num_layers - 1)], new_params - old_params

    def BP_update(self, x, y, gl_target):

        ## Get BP Gradients
        z = x.clone().detach()
        for i in range(0, self.num_layers - 1):
            z = self.wts[i](z)

        # Get loss
        if self.smax and self.crossEnt:
            loss = NLL(torch.log(softmax(z)), y.detach()) / z.size(0)  # CrossEntropy
        elif not self.smax and self.crossEnt:
            loss = bce(torch.sigmoid(z), gl_target.detach()) / z.size(0)
        else:
            loss = mse(softmax(z), gl_target.detach()) / z.size(0)
        self.bp_optim.zero_grad()
        loss.backward()

        # Perform update (either Adam or SGD)
        if self.adam:
            self.bp_optim.step()
        else:
            with torch.no_grad():
                for i in range(0, self.num_layers - 1):
                    self.wts[i][0].weight -= self.l_rate * self.wts[i][0].weight.grad



    def PC_update(self, targ, y):

        ## Update each weight matrix
        for i in range(self.num_layers - 1):

            # Compute local losses, sum neuron-wise and avg batch-wise
            if i < (self.num_layers - 2) or not self.crossEnt:
                p = self.wts[i](targ[i].detach())
                loss = .5 * mse(p, targ[i + 1].detach()) / p.size(0)
            elif self.smax:
                p = self.wts[i](targ[i].detach())
                # loss = NLL(torch.log(softmax(p)), y.detach()) / p.size(0)
                loss = -torch.mean((targ[-1].detach() * torch.log(softmax(p))).sum(1))
            else:
                p = self.wts[i](targ[i].detach())
                loss = bce(p, targ[i + 1].detach()) / p.size(0)

            # Compute normalized weight gradients
            self.optims[i].zero_grad()
            loss.backward()
            with torch.no_grad():
                if i == 2:
                    nm = torch.mean(torch.square(relu(targ[i])).sum(1)) + self.bias + self.eps
                    self.wts[i][-2].weight.grad /= nm
                    self.wts[i][-2].bias.grad /= nm
                else:
                    nm = torch.mean(torch.square(relu(targ[i])).sum(1)) + self.bias + self.eps
                    self.wts[i][-1].weight.grad /= nm
                    self.wts[i][-1].bias.grad /= nm

            self.optims[i].step()