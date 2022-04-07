# CS523_Project
Deep learning project 2022
Our task is to train a model to be both robust and deferentially private (DP) by using batch normalization can improve the trade-off observed in literature. 
## SIDP
This is pytorch Implementation of ' Robust Differentially Private Training of Deep Neural Networks' [[1]](#1)
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
| Y :|:   Y  :|: Y  :|:7.1|:   0.97 :|
|----|:-------|:----:|:--:|---------:|
| Y :|:   Y  :|: N  :|:7.1|:   0.94 :|


## References
<a id="1">[1]</a> 
Ali Davody, David Ifeoluwa Adelani, Thomas Klein-
bauer, and Dietrich Klakow. On the effect of normal-
ization layers on differentially private training of deep
neural networks. CoRR, abs/2006.10919, 2020.
