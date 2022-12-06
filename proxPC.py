import torch
from torch import nn
from utilities import relu_d

softmax = nn.Softmax(dim=1)
NLL = nn.NLLLoss(reduction='sum')
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()


class ProxPC(nn.Module):
    def __init__(self, layer_szs, lr=.001, n_iter=25, bot_gamma=.02, top_gamma=.02, t_type=2, eps=1, bias=True,
                 rand_init=False, crossEnt=True, RK=0, mom=False, smax=True, linear=False, normalize=True, dec_gamma=.00,
                 adam_lr=.00003):
        super().__init__()

        self.num_layers = len(layer_szs)
        self.layer_szs = layer_szs
        self.n_iter = n_iter
        self.bias= bias
        self.normalize = normalize
        self.RK = RK                          #RK=0: No RK,  RK=1: RK w/ continuous updates,  RK=2: RK w/ stored updates
        self.l_rate = lr
        self.linear = linear
        self.target_type = t_type
        self.adam_lr = adam_lr
        self.wts, self.optims = self.create_wts()
        self.bp_optim = torch.optim.Adam(self.wts.parameters(), lr=self.l_rate)
        self.bot_gamma = bot_gamma
        self.top_gamma = top_gamma
        self.dec_gamma = dec_gamma
        self.eps = eps
        self.rand_init = rand_init
        self.crossEnt = crossEnt
        self.mom = mom
        self.smax= smax




    def create_wts(self):

        w = nn.ModuleList([])
        w_optims = []

        for l in range(0, self.num_layers - 1):
            if self.linear:
                w.append(nn.Sequential(nn.Linear(self.layer_szs[l], self.layer_szs[l + 1], bias=self.bias)))
            else:
                w.append(nn.Sequential(nn.ReLU(), nn.Linear(self.layer_szs[l], self.layer_szs[l + 1], bias=self.bias)))
            #w_optims.append(torch.optim.SGD(w[-1].parameters(), lr=1, momentum=.9, dampening=.9))
            w_optims.append(torch.optim.Adam(w[-1].parameters(), lr=self.adam_lr))

        return nn.ModuleList(w), w_optims


    ############################## COMPUTE FORWARD VALUES ##################################
    def compute_values(self, x):
        with torch.no_grad():
            h = [torch.randn(1, 1) for i in range(self.num_layers)]

            #First h is the input
            h[0] = x.clone()

            #Compute all hs except last one.
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



    ############################## PC Optimized Activity ##################################
    def compute_PC_targets(self, h, global_target, y=None):
        with torch.no_grad():
            targ = [h[i].clone() for i in range(self.num_layers)]
            nm = torch.mean(torch.square(relu(h[-2])).sum(1)).item() + self.eps
            targ[-1] = (1 / (1 + nm*self.l_rate)) * softmax(h[-1]) + (nm*self.l_rate / (1 + nm*self.l_rate)) * global_target.clone()

            eps = [torch.zeros_like(h[i]) for i in range(self.num_layers)]
            p = [self.wts[i](targ[i].detach()) for i in range(self.num_layers-1)]
            p.insert(0, h[0].clone())

        # Iterative updates
        for i in range(self.n_iter):
            with torch.no_grad():
                #Compute hidden errors
                for layer in range(1, self.num_layers-1):
                    eps[layer] = targ[layer] - p[layer]                 #MSE gradient
                
                #Compute output layer
                if self.smax:
                    eps[-1] = (targ[-1] - softmax(p[-1])) #Cross-ent w/ softmax gradient
                elif self.crossEnt:
                    eps[-1] = (targ[-1] - torch.sigmoid(p[-1]))       #Binary cross-ent with sigmoid
                else:
                    eps[-1] = (targ[-1] - p[-1])                      #MSE


            #Update Targets
            for layer in range(1, self.num_layers - 1):
                with torch.no_grad():
                    dfdt = eps[layer + 1].matmul(self.wts[layer][1].weight) * relu_d(targ[layer])
                    dt = self.top_gamma * -eps[layer] + self.bot_gamma * dfdt
                    targ[layer] = targ[layer] + dt - self.dec_gamma * targ[layer]

            #Compute output layer target use learning rate and norm of presynaptic activity
            with torch.no_grad():
                nm = torch.mean(torch.square(relu(targ[-2])).sum()).item() + self.bias + self.eps
                targ[-1] = (1 / (1 + nm*self.l_rate)) * softmax(p[-1]) + (nm*self.l_rate / (1 + nm*self.l_rate)) * global_target.clone()

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
    def train_wts(self, x, global_target, y, get_grads=False):

        # Record params before update
        with torch.no_grad():
            old_wts = [self.wts[0][-1].weight.data.clone()]
            old_params = self.wts[0][-1].weight.data.clone().view(-1)
            for i in range(1, self.num_layers - 1):
                old_wts.append(self.wts[i][1].weight.data.clone())
                old_params = torch.cat((old_params, self.wts[i][-1].weight.data.clone().view(-1)), dim=0)

            # Get feedforward and target values
            if self.rand_init:
                h = self.compute_rand_values(x)
            else:
                h = self.compute_values(x)
            h_hat = self.compute_targets(h, global_target, y)


        # Update weights
        if self.target_type == 2 and self.RK == 0:
            self.PC_update(h_hat, y)
        elif self.target_type == 0:
            self.BP_update(x, y, global_target)
        elif self.target_type == 1:
            self.Impl_SGD_update(x, y, global_target)
        elif self.target_type == 3:
            self.normalized_BP_update(x, global_target, y)


        # Record new params
        with torch.no_grad():
            new_wts = [self.wts[0][-1].weight.data.clone()]
            new_params = self.wts[0][-1].weight.data.clone().view(-1)
            for i in range(1, self.num_layers - 1):
                new_wts.append(self.wts[i][-1].weight.data.clone())
                new_params = torch.cat((new_params, self.wts[i][-1].weight.data.clone().view(-1)), dim=0)

            #return h, change in individual weights, and change in param vector
            return h, h_hat, [new_wts[w] - old_wts[w] for w in range(self.num_layers-1)], new_params - old_params



    def BP_update(self, x, y, gl_target):

        ## Get BP Gradients
        z = x.clone().detach()
        for i in range(0, self.num_layers - 1):
            z = self.wts[i](z)

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
                    self.wts[i][1].weight -= self.l_rate * self.wts[i][1].weight.grad

                    

    def normalized_BP_update(self, x, gl_target, y):
        with torch.no_grad():
            h = self.compute_values(x)
            
            nm = torch.mean(torch.square(relu(h[-2])).sum(1)).item() + self.eps
            out_targ = (1 / (1 + nm*self.l_rate)) * softmax(h[-1]) + (nm*self.l_rate / (1 + nm*self.l_rate)) * gl_target.clone()
            #out_targ = self.n * gl_target.clone() + (1 - self.n) * softmax(h[-1])



        ## Get BP Gradients
        z = x.clone().detach()
        for i in range(0, self.num_layers - 1):
            z = self.wts[i](z)

        # Get loss
        if self.smax and self.crossEnt:
            #loss = NLL(torch.log(softmax(z)), y.detach()) / z.size(0)  # CrossEntropy
            loss = -torch.mean((out_targ.detach() * torch.log(softmax(z))).sum(1))
        elif not self.smax and self.crossEnt:
            loss = bce(torch.sigmoid(z), out_targ.detach()) / z.size(0)
        else:
            loss = .5 * mse(z, out_targ.detach()) / z.size(0)
        self.bp_optim.zero_grad()
        loss.backward()

        with torch.no_grad():
            for i in range(0, self.num_layers - 1):
                nm = torch.mean(torch.square(relu(h[i])).sum(1))
                self.wts[i][1].weight -= self.wts[i][1].weight.grad / (nm + self.eps)




    def Impl_SGD_update(self, x, y, gl_target):
        #Here we compute the implicit SGD update. We do so under the assumption alpha is large enough that
        # the loss (L) after the implicit update is near zero. (see proximal equation)

        with torch. no_grad():
            theta_b = [self.wts[w][1].weight.clone() for w in range(self.num_layers - 1)]
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
                for i in range(self.num_layers - 1):
                    self.wts[i][1].weight -= step_sz * (self.wts[i][1].weight.grad + inv_alpha * (self.wts[i][1].weight - theta_b[i]))

            self.bp_optim.zero_grad()



    def PC_update(self, targ, y, RK_upd=None):

        ## Update each weight matrix
        for i in range(self.num_layers-1):

            #Compute local losses, sum neuron-wise and avg batch-wise
            with torch.no_grad():
                #n = torch.square(relu(targ[i])).sum(1) + self.eps

                if i < (self.num_layers - 2) or not self.crossEnt:
                    p = self.wts[i](targ[i].detach())
                    #e = ((targ[i+1].detach() - p).t() / n).t()
                    e = targ[i+1].detach() - p
                    #loss = .5 * mse(p, targ[i+1].detach()) / p.size(0)
                elif self.smax:
                    p = self.wts[i](targ[i].detach())
                    #e = ((targ[i+1].detach() - softmax(p)).t() / n).t()
                    e = targ[i+1].detach() - softmax(p)
                    #loss = NLL(torch.log(softmax(p)), y.detach()) / p.size(0)
                else:
                    p = self.wts[i](targ[i].detach())
                    #e = ((targ[i+1].detach() - torch.sigmoid(p)).t() / n).t()
                    e = targ[i+1].detach() - torch.sigmoid(p)
                    #loss = bce(p, targ[i+1].detach()) / p.size(0)

            #Compute weight gradients
            self.optims[i].zero_grad()
            #loss.backward()
            with torch.no_grad():
                #nm = torch.mean(torch.square(relu(targ[i])).sum(1)) + self.bias + self.eps
                self.wts[i][-1].weight.grad = -e.t().matmul(relu(targ[i])) / (e.size(0))
                if self.bias:
                    self.wts[i][-1].bias.grad = torch.mean(-e, dim=0)


            #Update weights (either Adam or SGD)
            if self.mom:
                nm = torch.mean(torch.square(relu(targ[i])).sum(1)) + self.bias + self.eps
                self.wts[i][-1].weight.grad /= nm
                self.wts[i][-1].bias.grad /= nm
                self.optims[i].step()
            else:
                with torch.no_grad():
                    nm = torch.mean(torch.square(relu(targ[i])).sum(1)) + self.bias + self.eps
                    self.wts[i][-1].weight -= self.wts[i][-1].weight.grad / nm
                    self.wts[i][-1].bias -= self.wts[i][-1].bias.grad / nm
