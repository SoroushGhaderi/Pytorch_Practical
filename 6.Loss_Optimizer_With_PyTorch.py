import torch
import torch.nn as nn


# create arrays f = 3x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
Y = torch.tensor([[3], [6], [9], [12]], dtype=torch.float)

n_samples, n_features = X.shape
print("n_samples: {}, n_features: {}".format(n_samples, n_features))

# build 1*1 network, only one neuron
input_size = n_features
output_size = n_features


# define custom model
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


# main function
def main():
    # model hyper parameters
    learning_rate: float = 0.1
    n_iter: int = 60
    # torch pipeline
    model = LinearRegression(input_size, output_size)
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Iteration
    for epoch in range(n_iter):
        # prediction = forward pass
        y_pred = model(X)
        # loss
        loss_function = loss(Y, y_pred)
        # gradients = backward pass
        loss_function.backward()  # derivative of loss_function/dw
        # update parameters with optimizer
        optimizer.step()
        # zero gradients
        optimizer.zero_grad()
        # print network summary
        if epoch % 2 == 0:
            [w, b] = model.parameters()
            print("loss: {loss_value:.3f}, "
                  "epoch: {nth_epoch} "
                  "weight: {nth_weight:.3f}"
                  .format(loss_value=loss_function,
                          nth_epoch=epoch + 1,
                          nth_weight=w[0][0]))


if __name__ == '__main__':
    main()
