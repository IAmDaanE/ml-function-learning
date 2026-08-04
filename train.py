import matplotlib.pyplot as plt
import os
import nnlib as nn 
import numpy as np

hidden_size = 160

network = nn.Network(nn.Losses.mse, 1, hidden_size, 3, 1, 1920, 1080)

network.add(nn.Layer(1, hidden_size, nn.Activations().relu))
network.add(nn.Layer(hidden_size, hidden_size, nn.Activations().relu))
network.add(nn.Layer(hidden_size, hidden_size, nn.Activations().relu))
network.add(nn.Layer(hidden_size, 1, nn.Activations().linear))

plt.ion()

def run_training(epochs, start_lr):
    current_lr = start_lr

    x = np.linspace(-200, 200, 1000).reshape(-1, 1)
    correct_y = np.sin(x / 15) * 10 # the actual function to learn

    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = correct_y.mean(), correct_y.std()

    x_norm = (x - x_mean) / x_std
    y_norm = (correct_y - y_mean) / y_std

    for epoch in range(epochs):
        predicted_y_norm = network.forward(x_norm)
        loss = network.loss_function(predicted_y_norm, y_norm)
        network.loss = loss
        network.backward(predicted_y_norm, y_norm)
        network.update(current_lr)
        current_lr = nn.LrDecays.linear_decay(current_lr, 0.000095, 0.0005)
        if epoch % 10 == 0:
            predicted_y = predicted_y_norm * y_std + y_mean
            plt.clf()
            plt.plot(x, correct_y, label='true')
            plt.plot(x, predicted_y, label='predicted')
            plt.title(f'epoch {epoch} | loss {loss:.6f}')
            plt.legend()
            plt.pause(0.01)
            network.visualize()

if __name__ == "__main__":
    run_training(50000, 0.1)