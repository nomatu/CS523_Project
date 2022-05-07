# CS523_Project
Deep learning Team Project 2022 - The tradeoff between privacy and robustness when training with Batch Normalization

Our task is to train a model to be both robust and deferentially private (DP) with and without using Batch Normalization. We want to test whether batch normalization can improve the trade-offs between robustness and privacy observed in the literature. We train models for two image classification tasks: MNIST and CIFAR-10. 

## Usage

To run experiments on MNIST use: 
```
python vision.py --dataset mnist 
```
To run experiments on CIFAR-10 use: 
```
python vision.py --dataset cifar10
```
Check ``vision.py`` for all other parameters that can be specified, such as number of epochs, privacy parameters, robustness parameters, etc. 
Check ``images`` file for all results of training on the MNIST and CIFAR-10 datasets with Projected Gradient Descent and ``src`` for all the files used for the project such as  


We have combined the DP with Batch Norm training algorithm of [[1]](#1) together with robust Projected Gradient Descent (PGD) training. We have updated the [code](https://github.com/uds-lsv/SIDP) of [[1]](#1) to support robust training. For each batch, we create a new adversarial batch formed by doing gradient ascent on the current batch. The goal of gradient ascent is to maximize the loss the model on the perturbed batch. The [following code](https://gist.github.com/oscarknagg/45b187c236c6262b1c4bbe2d0920ded6##file-projected_gradient_descent-py) for PGD is obtained and modified to work for our traiing procedure. 

To preform BatchNorm, the approach of [[1]](#1) is to use a small public dataset and augment each batch of the data with the public dataset. The public dataset is disjoint from the training data. The Public dataset does not contribute to training, but is only used to calculate the mean and standard deviation for each normalization layer. For MNIST, the apporach of [[1]](#1) is to use 128 image form to KMIST datset as the publicly available dataset.

## Testing
We compare DP + robust training on the MNIST and CIFAR-10 datasets (with and without BatchNorm) for various amounts of adversarial noise and privacy budgets. The comparison will be in terms of the accuracy acheived by the model on the test dataset.

## Results
The following experiments were run with the model for the MNIST classification task for 10 epochs and the CIFAR-10 classification for 20 epochs: 





## References
<a id="1">[1]</a> 
Ali Davody, David Ifeoluwa Adelani, Thomas Klein-
bauer, and Dietrich Klakow. On the effect of normalization layers on differentially private training of deep
neural networks. CoRR, abs/2006.10919, 2020.
