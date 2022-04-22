import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, RandomSampler
import torchvision.transforms as transforms
from tqdm import tqdm
from .dataset import data_loaders, axi_loader
from src.pgd_attack import projected_gradient_descent, viz_adversarial

import tensorflow as tf
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib


def train(model, criterion, optimizer, device, train_loader, clip, noise_multiplier, private, batch_size, axi_x, robust, adv_step_size, adv_alpha, adv_num_steps):
    model.train()
    loss_fn_robust = torch.nn.CrossEntropyLoss()
    for x, y in tqdm(train_loader):
        _lr = optimizer.param_groups[0]['lr']
        if private:
          std_params = _lr * clip * noise_multiplier / np.sqrt(batch_size)
        else:
          std_params = 0.0

        x_adv = x
        
        if robust:
          x_adv = projected_gradient_descent( model, x, y, loss_fn = loss_fn_robust, std_params = std_params, 
                                              axi_x = axi_x, device = device, step_size = adv_step_size, 
                                              num_steps = adv_num_steps, alpha = adv_alpha) 
        
       
        #viz_adversarial(x_adv, dataset = "mnist")
        x = torch.cat([x_adv.to(device), axi_x], dim=0) 
        y = y.to(device)

        optimizer.zero_grad()
        output = model.forward(x, std_params)[:batch_size]
        losses = criterion(output, y)

        saved_var = dict()
        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name] = torch.zeros_like(tensor)

        for j in losses:
            j.backward(retain_graph=True)
            if private:
              torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for tensor_name, tensor in model.named_parameters():
                new_grad = tensor.grad
                saved_var[tensor_name].add_(new_grad)
            optimizer.zero_grad()

        for tensor_name, tensor in model.named_parameters():
            tensor.grad = saved_var[tensor_name] / losses.shape[0]

        optimizer.step()


def evaluate_accuracy(model, data_loader, device, axi_x):
    model.eval()
    correct = 0
    total = 0.
    with torch.no_grad():
        for x, y in data_loader:
            _batch_size = x.shape[0]
            x = torch.cat([x.to(device), axi_x], dim=0)
            output = model(x, 0)[:_batch_size]
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.to(device).view_as(pred)).sum().item()
            total += _batch_size
    accuracy = correct / total
    return accuracy

def main(noise_multiplier, clip, private, lr, batch_size, epochs, normalization_type, device, 
          robust, adv_step_size, adv_alpha, adv_num_steps, resume_at_epoch):
    from .model import LeNet5

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    model = LeNet5(normalization_type)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9, last_epoch=-1)

    train_loader, test_loader = data_loaders(batch_size)
    axi_x = axi_loader(30).to(device)

    for epoch in range(epochs):
        
        train(model, criterion, optimizer, device, train_loader, clip, noise_multiplier, private, 
              batch_size, axi_x, robust, adv_step_size, adv_alpha, adv_num_steps)
        scheduler_lr.step()
        test_accuracy = evaluate_accuracy(model, test_loader, device, axi_x)

        print('epoch', epoch)
        print('valid accuracy  ', test_accuracy)
        eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(60000, batch_size, noise_multiplier, epoch+1, 1e-5)
        #print('For delta=1e-5, the current epsilon is: %.2f' % eps)
        print('---------------------')











