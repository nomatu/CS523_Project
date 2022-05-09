# CS523_Project
Deep learning Team Project 2022 - The tradeoff between privacy and robustness when training with Batch Normalization

Our task is to train a model to be both robust and deferentially private (DP) with and without using Batch Normalization. We want to test whether batch normalization can improve the trade-offs between robustness and privacy observed in the literature. We train models for two image classification tasks: MNIST and CIFAR-10. 

We have combined the DP with Batch Norm training algorithm of [[1]](#1) together with robust Projected Gradient Descent (PGD) training. We have updated the [code](https://github.com/uds-lsv/SIDP) of [[1]](#1) to support robust training. For each batch, we create a new adversarial batch formed by doing gradient ascent on the current batch. The goal of gradient ascent is to maximize the loss the model on the perturbed batch. The [following code](https://gist.github.com/oscarknagg/45b187c236c6262b1c4bbe2d0920ded6##file-projected_gradient_descent-py) for PGD is obtained and modified to work for our training procedure. 

To perform BatchNorm, the approach of [[1]](#1) is to use a small public dataset and augment each batch of the data with the public dataset. The public dataset is disjoint from the training data. The Public dataset does not contribute to training, but is only used to calculate the mean and standard deviation for each normalization layer. For MNIST, the apporach of [[1]](#1) is to use 128 image form to KMIST datset as the publicly available dataset.

## Usage

You will need to install the tensorflow-privacy package 

```
pip install tensorflow_privacy
```

To run experiments on MNIST use: 
```
python vision.py --dataset mnist 
```
To run experiments on CIFAR-10 use: 
```
python vision.py --dataset cifar10
```
Check ``vision.py`` for all other parameters that can be specified, such as number of epochs, privacy parameters, robustness parameters, etc. 

The ``images`` folder contains graphs from our experimental results. 

The ``src`` folder contains all the files needed to conduct the experiments. In the ``cifar10`` and ``mnist`` folder, the ``dataset.py`` file prepares and loads the datasets, the ``model.py`` file specifies the model, and the ``train.py`` file specifies the training procedure. 

The ``dp-layers`` folder contains DP specifications of layers used to build the CIFAR-10 and MNIST models. 

The ``pgd_attack.py`` file is used to perform the PGD attack for robust training. 


## Testing
We compare the performance of DP+robust training on classification tasks for MNIST and CIFAR-10 between the case when the model uses BatchNorm layers vs. when it does not.  We will compare the role of the addition of BatchNorm for different values of the noise multiplier. To that end we calculate the accuracy on the test set between a model trained with BatchNorm vs without BatchNorm, which are trained 
- with the same privacy parameters
- for the same number of epochs     
- the same robustness parameters

The accuracy is the percentage of images in the test set for which the model outputs the correct label. Each accuracy value is averaged over three independent runs. Since the privacy parameters and number of epochs are the same, this means that the models will use the same privacy budgets. The privacy budget is calculated as a function of the noise multiplier, number of epochs, and batch size. 

We tune the learning rate for each trained model.

## Results
The following experiments were run with the LeNet-5model for the MNIST classification task and the ResNet-18 model the CIFAR-10 classification task.

![Test Image 2](images/MNIST-1.png)
**Figure 1** depicts the test set accuracy of the LeNet-5 MNIST model for different values of privacy budget.  (The accuracy is averaged over 3 independent runs). Each dot in the graphs represents one epoch of training. All models are trained for 10 epochs. We repeat the experiments with different values of noise multipliers. We observe that the use of BatchNorm consistently improves the performance of the model for all noise multipliers. The difference in accuracy between using BatchNorm vs no BatchNorm is as high as 5 percentage points for noise multiplier 10 and eps = 0.01 and noise multiplier = 5 and eps = 0.02. For lower amounts of noise and higher privacy budgets the difference in accuracy is around 1 percentage point. 

![Test Image 2](images/CIFAR10-1.png)
**Figure 2** depicts the average test set accuracy of the CIFAR-10 model for different values of privacy budget. The models are trained for 20 epochs. More epochs would have been needed for the models to achieve reasonable accuracy values. However, robust and DP training is extremely slow for the ResNet-18 model, with one epoch of training lasting about 1 hour. For the first 20 epochs of training, we do not observe a significant difference in the accuracy obtained with BatchNorm vs without it. A distinction between the two methods could potentially show up in much later epochs of training. As expected, the accuracy achieved by the models decreases with the increase in the noise multiplier. 

The parameters for both models are as following: 
- LeNet-5 : Learning rate = 0.01 except for when noise multiplier = 1, where learning rate = 0.1; gradient clipping value = 3.1; batch size = 32 [[1]](#1); robust adversarial steps = 40; 
- ResNet-18: Learning rate = 0.01 for all noise multipliers; gradient clipping value = 2.5; batch size = 32 [[1]](#1); robust adversarial steps = 20 
- For both models we use the following parameters for the PGD attack: gradient step = 0.01 and projection norm = 0.03 [[2]](#2);

## References
<a id="1">[1]</a> 
Ali Davody, David Ifeoluwa Adelani, Thomas Klein-
bauer, and Dietrich Klakow. On the effect of normalization layers on differentially private training of deep
neural networks. CoRR, abs/2006.10919, 2020.

<a id="2">[2]</a> 
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. To-wards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR), 2018
