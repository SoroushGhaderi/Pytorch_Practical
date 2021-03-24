import torch
import torch.nn as nn


# create arrays f = 3x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float)
Y = torch.tensor([3, 6, 9, 12], dtype=torch.float)
# weight initialization
w = torch.tensor(0.1, dtype=torch.float, requires_grad=True)


# forward propagation step
def forward(x):
    return torch.mul(x, w)


# main function
def main():
    learning_rate: float = 0.01
    n_iter: int = 40
    global w

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD([w], lr=learning_rate)
    # Iteration
    for epoch in range(n_iter):
        # prediction = forward pass
        y_pred = forward(X)
        # loss
        loss_function = loss(Y, y_pred)
        # gradients = backward pass
        loss_function.backward()  # dloss_function/dw
        optimizer.step()
        # zero gradients
        optimizer.zero_grad()
        # print network summary
        if epoch % 2 == 0:
            print("loss: {loss_value}, epoch: {nth_epoch} weight: {nth_weight}"
                  .format(loss_value=loss_function, nth_epoch=epoch + 1, nth_weight=w))


if __name__ == '__main__':
    main()
