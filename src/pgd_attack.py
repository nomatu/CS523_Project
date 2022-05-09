import torch
import matplotlib.pyplot as plt
import numpy as np

def projected_gradient_descent(model, x, y, loss_fn, std_params, axi_x, 
                               num_steps, step_size,  alpha, 
                               clamp=(0,1), device="cuda:0", dataset = "mnist"):
    """Performs the projected gradient descent attack on a batch of images."""
    x_adv = x.clone().detach().requires_grad_(True).to(device)
    x = x.to(device)
    y = y.to(device)
    num_channels = x.shape[1]

    for i in range(num_steps):

        _x_adv = x_adv.clone().detach().requires_grad_(True).to(device)

        _x_adv_augmented = torch.cat([_x_adv.to(device), axi_x], dim=0) 
        
        batch_size = x.shape[0]

        prediction = model(_x_adv_augmented, std_params)[:batch_size]
        
        y_hat = torch.argmax(prediction, dim = 1)
        loss = loss_fn(prediction, y)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a l_infty norm
            gradients = _x_adv.grad.sign() * step_size
            x_adv += gradients

        # Project back into l_infty ball and correct range
        x_adv = torch.max(torch.min(x_adv, x + alpha), x - alpha)
        x_adv = x_adv.clamp(*clamp)
        
    return x_adv.detach()

def viz_adversarial(x, y_hat = None, dataset="mnist"):
  x = x.clone().cpu().detach().numpy()
  if dataset == "mnist":
    plt.imshow(x[0, 0], cmap='gray')
  elif dataset == "cifar10":
    img = x[2]
    img = img * 0.2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))
  
  import datetime
  ts = datetime.datetime.now().timestamp()
  filename = "images/" + str(ts) + ".png"
  plt.savefig(filename)
  
  if y_hat != None: plt.title("predction = {}".format(y_hat[0].item()))
