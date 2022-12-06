import train_MLP as tmlp
import train_AE as tae



'''Each line in the main function execute training runs over 5 seeds for one algorithm in one task.
   These are preset to use the same hyperparameters that were used in the original paper.
   However, the training_run function will automatically compute a grid search over learning rates if more than one learning
   rate is entered to the the argument lr=[lr1, lr2,...lrN] and save_best is set to True. The data from the 5 seeds 
   that had the best peforming learning rate will be saved. The data is saved to the 'data' folder. The plot script may 
   then be used to summarize and plot the data.
   '''
def main():
    #Stability Test MNIST
    tmlp.training_run(save_best=False, max_iters=20000, data=0, batch_size=1, num_seeds=5, model_type=0,lrs=[.01, .1, 1, 2.5, 10, 100])
    # tmlp.training_run(save_best=False, max_iters=20000, data=0, batch_size=1, num_seeds=5, model_type=3,lrs=[.01, .1, 1, 2.5, 10, 100])
    # tmlp.training_run(save_best=False, max_iters=20000, data=0, batch_size=1, num_seeds=5, model_type=6, lrs=[.01, .1, 1, 2.5, 10, 100], eps=0)
    # tmlp.training_run(save_best=False, max_iters=20000, data=0, batch_size=1, num_seeds=5, model_type=8, lrs=[.01, .1, 1, 2.5, 10, 100], eps=0)

    # Stability Test CIFAR-10
    # tmlp.training_run(save_best=False, max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=0, lrs=[.01, .1, 1, 2.5, 10, 100])
    # tmlp.training_run(save_best=False, max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=3, lrs=[.01, .1, 1, 2.5, 10, 100])
    # tmlp.training_run(save_best=False, max_iters=50000, data=2, batch_size=1, num_seeds=1, model_type=6, lrs=[.01, .1, 1, 2.5, 10, 100], eps=0)
    # tmlp.training_run(save_best=False, max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=8, lrs=[.01, .1, 1, 2.5, 10, 100], eps=0)

    # CIFAR-10 mini-batch 1
    # tmlp.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=0, lrs=[.015])
    # tmlp.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=1, lrs=[.000025])
    # tmlp.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=3, lrs=[.01])
    # tmlp.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=6, lrs=[100], eps=1)
    # tmlp.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=7, lrs=[100], eps=5)
    # tmlp.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=9, lrs=[.00001], n=.95)
    # tmlp.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=10, lrs=[100], eps=5)

    # CIFAR-10 mini-batch 64
    # tmlp.training_run(max_iters=77000, data=2, batch_size=64, num_seeds=5, model_type=0, lrs=[.01])
    # tmlp.training_run(max_iters=77000, data=2, batch_size=64, num_seeds=5, model_type=1, lrs=[.00001])
    # tmlp.training_run(max_iters=77000, data=2, batch_size=64, num_seeds=5, model_type=3, lrs=[.01])
    # tmlp.training_run(max_iters=77000, data=2, batch_size=64, num_seeds=5, model_type=6, lrs=[100], eps=5)
    # tmlp.training_run(max_iters=77000, data=2, batch_size=64, num_seeds=5, model_type=7, lrs=[100], eps=5)
    # tmlp.training_run(max_iters=77000, data=2, batch_size=64, num_seeds=5, model_type=9, lrs=[.000005], n=.35)
    # tmlp.training_run(max_iters=77000, data=2, batch_size=64, num_seeds=5, model_type=10, lrs=[100], eps=5)

    # MNIST mini-batch 1
    # tmlp.training_run(max_iters=50000, data=0, batch_size=1, num_seeds=5, model_type=0, lrs=[.015])
    # tmlp.training_run(max_iters=50000, data=0, batch_size=1, num_seeds=5, model_type=1, lrs=[.0001])
    # tmlp.training_run(max_iters=50000, data=0, batch_size=1, num_seeds=5, model_type=3, lrs=[.05])
    # tmlp.training_run(max_iters=50000, data=0, batch_size=1, num_seeds=5, model_type=6, lrs=[5], eps=.25)
    # tmlp.training_run(max_iters=50000, data=0, batch_size=1, num_seeds=5, model_type=7, lrs=[3], eps=.25)

    # Fashion-MNIST mini-batch 1
    # tmlp.training_run(max_iters=50000, data=0, batch_size=1, num_seeds=5, model_type=0, lrs=[.01])
    # tmlp.training_run(max_iters=50000, data=0, batch_size=1, num_seeds=5, model_type=1, lrs=[.00005])
    # tmlp.training_run(max_iters=50000, data=0, batch_size=1, num_seeds=5, model_type=3, lrs=[.03])
    # tmlp.training_run(max_iters=50000, data=0, batch_size=1, num_seeds=5, model_type=6, lrs=[2.5], eps=.25)
    # tmlp.training_run(max_iters=50000, data=0, batch_size=1, num_seeds=5, model_type=7, lrs=[2.5], eps=.25)

    # CIFAR-10 Autoencoder
    # tae.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=0, lrs=[.0001])
    # tae.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=1, lrs=[.00005])
    # tae.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=3, lrs=[.04])
    # tae.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=6, lrs=[10], eps=20)
    # tae.training_run(max_iters=50000, data=2, batch_size=1, num_seeds=5, model_type=7, lrs=[10], eps=20)

    # CIFAR-10 Convolutional
    # training_run(max_iters=77000, batch_size=64, num_seeds=5, n=1, smax=True, lrs=[.00002], model_type=1, save_best=True)
    # training_run(max_iters=77000, batch_size=64, num_seeds=5, n=1, smax=True, lrs=[.000009], model_type=3, save_best=True)
    # training_run(max_iters=77000, batch_size=64, num_seeds=5, smax=True, lrs=[.000008], model_type=4, save_best=True, eps=15)


