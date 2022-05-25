import torch
from torch import nn
from utilities import relu_d
from utilities import tanh_d

softmax = nn.Softmax(dim=1)
NLL = nn.NLLLoss(reduction='sum')
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()

class PC(nn.Module):
    def __init__(self, layer_szs, lr=.001, func=nn.Tanh(), t_type=0, n_iter=25, bot_gamma=.07, top_gamma=.07, n=1,
                 rand_init=False, crossEnt=True, RK=0, adam=False, smax=True, linear=False, normalize=False, dec_gamma=.0,
                 bias=True):
        super().__init__()

        self.num_layers = len(layer_szs)
        self.layer_szs = layer_szs
        self.n_iter = n_iter
        self.normalize = normalize
        self.adam = adam
        self.l_rate = lr
        self.target_type = t_type  # 0=BP, 2=IL
        self.bias = bias
        self.func = func
        self.linear = linear
        self.wts, self.optims = self.create_wts()
        self.bp_optim = torch.optim.Adam(self.wts.parameters(), lr=self.l_rate)
        self.bot_gamma = bot_gamma
        self.top_gamma = top_gamma
        self.dec_gamma = dec_gamma
        self.n = n
        self.rand_init = rand_init
        self.crossEnt = crossEnt
        self.smax= smax


    def create_wts(self):

        w = nn.ModuleList([])
        w_optims = []

        for l in range(self.num_layers - 2):
            if self.linear:
                w.append(nn.Sequential(nn.Linear(self.layer_szs[l], self.layer_szs[l + 1], bias=self.bias)))
            else:
                w.append(nn.Sequential(nn.Linear(self.layer_szs[l], self.layer_szs[l + 1], bias=self.bias), nn.ReLU()))
            if self.adam:
                w_optims.append(torch.optim.Adam(w[-1].parameters(), lr=self.l_rate))
            else:
                w_optims.append(torch.optim.SGD(w[-1].parameters(), lr=self.l_rate))

        w.append(nn.Sequential(nn.Linear(self.layer_szs[-2], self.layer_szs[-1], bias=self.bias)))
        if self.adam:
            w_optims.append(torch.optim.Adam(w[-1].parameters(), lr=self.l_rate))
        else:
            w_optims.append(torch.optim.SGD(w[-1].parameters(), lr=self.l_rate))

        return nn.ModuleList(w), w_optims


    ############################## COMPUTE INITIAL VALUES ##################################
    def compute_values(self, x):
        with torch.no_grad():
            h = [torch.randn(1, 1) for i in range(self.num_layers)]

            #First h is the input
            h[0] = x.clone()

            #Compute all hs except last one. Use Tanh as in paper
            for i in range(1, self.num_layers):
                h[i] = self.wts[i-1](h[i-1].detach())
        return h


    def compute_rand_values(self, x):
        with torch.no_grad():
            h = [torch.randn(x.size(0), self.layer_szs[i]) for i in range(len(self.layer_szs))]

            #First h is the input
            h[0] = x.clone()
            h[-1] = self.wts[-1](h[-2])

        return h



    ############################## PC TARGETS ##################################
    def compute_PC_targets(self, h, global_target, y=None):
        with torch.no_grad():
            targ = [h[i].clone() for i in range(self.num_layers)]
            targ[-1] = self.n * global_target.clone() + (1 - self.n) * softmax(h[-1])

            eps = [torch.zeros_like(h[i]) for i in range(self.num_layers)]
            p = [self.wts[i](targ[i].detach()) for i in range(self.num_layers-1)]
            p.insert(0, h[0].clone())


        # Iterative updates
        for i in range(self.n_iter):
            self.bp_optim.zero_grad()

            #Compute errors
            with torch.no_grad():
                for layer in range(1, self.num_layers-1):
                    eps[layer] = (targ[layer] - p[layer])                   #MSE gradient

                if self.smax:
                    eps[-1] = (targ[-1] - softmax(p[-1])) * .5           #Cross-ent w/ softmax gradient
                elif self.crossEnt:
                    eps[-1] = (targ[-1] - torch.sigmoid(p[-1]))       #Binary cross-ent with sigmoid
                else:
                    eps[-1] = (targ[-1] - p[-1])                      #MSE



            #Update Targets
            for layer in range(1, self.num_layers - 1):
                with torch.no_grad():
                    if layer < self.num_layers - 2:
                        dfdt = (eps[layer + 1] * relu_d(p[layer + 1])).matmul(self.wts[layer][0].weight)
                    else:
                        dfdt = (eps[layer + 1]).matmul(self.wts[layer][0].weight)
                    dt = self.top_gamma * -eps[layer] + self.bot_gamma * dfdt
                    targ[layer] = targ[layer] + dt - self.dec_gamma * targ[layer]

            with torch.no_grad():
                targ[-1] = self.n * global_target + (1-self.n) * softmax(p[-1])

            #Compute new Predictions
            with torch.no_grad():
                for layer in range(1, self.num_layers):
                    p[layer] = self.wts[layer-1](targ[layer-1].detach())


        return targ


    ############################## GENERAL TARGET PROP FUNCTION ##################################
    def compute_targets(self, h, global_target, y=None):
        if self.target_type == 2:
            targets = self.compute_PC_targets(h, global_target, y=y)
        else:
            targets = None

        return targets


    ############################## TRAIN ##################################

    def train_wts(self, x, global_target, y, h_old=None):

        # Record params before update
        with torch.no_grad():
            old_wts = [self.wts[0][0].weight.data.clone()]
            old_params = self.wts[0][0].weight.data.clone().view(-1)
            for i in range(1, self.num_layers - 1):
                old_wts.append(self.wts[i][0].weight.data.clone())
                old_params = torch.cat((old_params, self.wts[i][0].weight.data.clone().view(-1)), dim=0)

        # Get feedforward and target values
        if self.rand_init:
            h = self.compute_rand_values(x.detach())
        else:
            h = self.compute_values(x.detach())
        h_hat = self.compute_targets(h, global_target, y)


        # Update weights
        if self.target_type == 2:
            self.PC_update(h_hat, y)
        elif self.target_type == 0:
            self.BP_update(x, y, global_target)
        elif self.target_type == 1:
            self.Impl_SGD_update(x, y, global_target)

        # Record new params
        with torch.no_grad():
            new_wts = [self.wts[0][0].weight.data.clone()]
            new_params = self.wts[0][0].weight.data.clone().view(-1)
            for i in range(1, self.num_layers - 1):
                new_wts.append(self.wts[i][0].weight.data.clone())
                new_params = torch.cat((new_params, self.wts[i][0].weight.data.clone().view(-1)), dim=0)

        #return h, change in individual weights, and change in param vector
        return h, h_hat, [new_wts[w] - old_wts[w] for w in range(self.num_layers-1)], new_params - old_params



    def BP_update(self, x, y, gl_target):

        ## Get FF values
        z = x.clone().detach()
        for n in range(self.num_layers - 1):
            z = self.wts[n](z)

        #Get loss
        if self.smax and self.crossEnt:
            loss = NLL(torch.log(softmax(z)), y.detach()) / z.size(0) # CrossEntropy
        elif not self.smax and self.crossEnt:
            loss = bce(torch.sigmoid(z), gl_target.detach()) / z.size(0)
        else:
            loss = .5 * mse(z, gl_target.detach()) / z.size(0)
        self.bp_optim.zero_grad()
        loss.backward()


        # Perform update (either Adam or SGD)
        if self.adam:
            self.bp_optim.step()
        else:
            with torch.no_grad():
                for i in range(0, self.num_layers - 1):
                    self.wts[i][0].weight -= self.l_rate * self.wts[i][0].weight.grad



    '''def Impl_SGD_update(self, x, y, gl_target):

        with torch. no_grad():
            theta_b = [self.wts[w][0].weight.clone() for w in range(self.num_layers - 1)]
            inv_alpha = 1
            step_sz = .005

        for iter in range(500):
            # Get BP Gradients
            z = x.clone().detach()
            for n in range(0, self.num_layers - 1):
                z = self.wts[n](z)

            # Get loss
            if self.smax and self.crossEnt:
                loss = NLL(torch.log(softmax(z)), y.detach()) / z.size(0)  # CrossEntropy
            elif not self.smax and self.crossEnt:
                loss = bce(torch.sigmoid(z), gl_target.detach()) / z.size(0)
            else:
                loss = .5 * mse(z, gl_target.detach()) / z.size(0)
            self.bp_optim.zero_grad()
            loss.backward()

            # Perform update (either Adam or SGD)
            with torch.no_grad():
                #print(iter, loss)
                for i in range(self.num_layers - 1):
                    self.wts[i][0].weight -= step_sz * (self.wts[i][0].weight.grad + inv_alpha * (self.wts[i][0].weight - theta_b[i]))

            self.bp_optim.zero_grad()'''



    def PC_update(self, targ, y, RK_upd=None):

        ## Update each weight matrix
        for i in range(self.num_layers-1):

            #Compute local losses, sum neuron-wise and avg batch-wise
            if i < (self.num_layers - 2) or not self.crossEnt:
                p = self.wts[i](targ[i].detach())
                loss = .5 * mse(p, targ[i+1].detach()) / p.size(0)
            elif self.smax:
                p = self.wts[i](targ[i].detach())
                loss = -torch.mean((targ[-1].detach() * torch.log(softmax(p))).sum(1))
            else:
                p = self.wts[i](targ[i].detach())
                loss = bce(torch.sigmoid(p), targ[i+1].detach()) / p.size(0)

            #Compute weight gradients
            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
