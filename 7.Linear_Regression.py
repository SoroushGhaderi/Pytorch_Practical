import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def main():
    # create dataset
    X_numpy, y_numpy = datasets.make_regression(n_samples=100,
                                                n_features=1,
                                                noise=20,
                                                random_state=42)
    X = torch.from_numpy(X_numpy.astype(np.float32))
    y = torch.from_numpy(y_numpy.astype(np.float32))
    y = y.view(y.shape[0], 1)

    n_samples, n_features = X.shape
    input_size = n_features
    output_size = 1
    # model hyper parameters
    learning_rate: float = 0.1
    n_iter: int = 60
    # torch pipeline
    model = nn.Linear(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Iteration
    for epoch in range(n_iter):
        # prediction = forward pass
        y_pred = model(X)
        # loss
        loss = criterion(y, y_pred)
        # gradients = backward pass
        loss.backward()  # derivative of loss_function/dw
        # update parameters with optimizer
        optimizer.step()
        # zero gradients
        optimizer.zero_grad()
        # print network summary
        if epoch % 2 == 0:
            [w, _] = model.parameters()
            print("loss: {loss_value:.3f}, "
                  "epoch: {nth_epoch} "
                  "weight: {nth_weight:.3f}"
                  .format(loss_value=loss,
                          nth_epoch=epoch + 1,
                          nth_weight=w[0][0]))

    # define function for plot regression line
    def plot_regression(x_value, y_value):
        predicted = model(x_value).detach().numpy()
        # plot scatter of x and y
        plt.plot(x_value, y_value, "ro")
        # plot x and predicted value
        plt.plot(x_value, predicted, "b")
        plt.show()

    plot_regression(X, y)


if __name__ == '__main__':
    main()
