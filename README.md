# CS523_Project
Deep learning project 2022
Our task is to train a model to be both robust and DP by using batch normalization can improve the trade-off observed in literature. 
# SIDP
This is pytorch Implementation of ' Robust Differentially Private Training of Deep Neural Networks'
# Tessting

# Results
To preform BatchNorm, the approach of is useing a small public dataset and augment each batch of the data with the small public dataset disjoint from MNIST. The Public dataset does not contribute to training, but is only used to calculate the mean and standard deviation for each normalization layer. For MNIST, the apporach of is to use 128 image form to KMIST datset as  the publicly available dataset.
The following were run with the model for MNIST dataset with 1 epoch: 

1. DP and Robust - no normalization
2. DP and Robust - with normalization 
3. Robust (not DP) - with normalization
4. Robust (not DP) - no normalization
For 1 and 2, different values for noise-multiplier (the current default value is 7.1)
