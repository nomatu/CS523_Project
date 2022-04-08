# CS523_Project
Deep learning Team Project 2022 - The Tradeoff between privacy and robustness when training with Batch Normalization
Our task is to train a model to be both robust and deferentially private (DP) by using batch normalization can improve the trade-off observed in literature. 
## Usage
We combined the DP with Batch Norm training algorithm of [[1]](#1) together with robust PGD training. We have updated the [code](https://github.com/uds-lsv/SIDP) of [[1]](#1) to support robust training and changed the training procedure. For each batch, we created a new adversarial batch formed by doing gradient ascent on the current batch. The goal of gradient ascent is to maximize the loss the model on the perturbed batch. The code for [gradient](https://gist.github.com/oscarknagg/45b187c236c6262b1c4bbe2d0920ded6##file-projected_gradient_descent-py) is obtained and modified in a combatible way with the gradient update of [[1]](#1).
## Testing
We compared DP + robust training on the MNIST and CIFAR-10 datasets (with and without BatchNorm) for various amount of adversarial noise and privcy budgets. The comparison will be in terms of the accuracy acheived by the model on the test dataset.

## Results
To preform BatchNorm, the approach of[[1]](#1) is using a small public dataset and augment each batch of the data with the small public dataset disjoint from MNIST. The Public dataset does not contribute to training, but is only used to calculate the mean and standard deviation for each normalization layer. For MNIST, the apporach of [[1]](#1) is to use 128 image form to KMIST datset as  the publicly available dataset.
The following were run with the model for MNIST dataset with 1 epoch: 

1. DP and Robust - no normalization
2. DP and Robust - with normalization 

Different values for noise-multiplier (the current default value is 7.1).
The accuracy is calculated on the test set.
| DP | Robust | Norm | NM | Accuracy |
|----|:-------|:----:|:--:|---------:|
| Y  |    Y   |  Y   | 7.1|    0.97  |
| Y  |    Y   |  N   | 7.1|    0.94  |


## References
<a id="1">[1]</a> 
Ali Davody, David Ifeoluwa Adelani, Thomas Klein-
bauer, and Dietrich Klakow. On the effect of normalization layers on differentially private training of deep
neural networks. CoRR, abs/2006.10919, 2020.
