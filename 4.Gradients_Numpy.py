import numpy as np

# create arrays f = 3x
X = np.array([1, 2, 3, 4])
Y = np.array([3, 6, 9, 12])
# weight initialization
w: float = 0.1


# forward propagation step
def forward(x):
    return x * w


# define loss function
def loss(y, y_pred):
    return np.mean(np.square(y - y_pred))


# J = MSE = 1/n * ((w * X) - y) ** 2
# dJ/dw = 1/n * 2x * ((w * X) - y)
def gradient(x, y, y_pred):
    return np.mean(np.dot(2 * x, (y_pred - y)))


# main function
def main():
    learning_rate: float = 0.01
    n_iter: int = 20
    global w
    # Iteration
    for epoch in range(n_iter):
        # forward
        y_pred = forward(X)
        # loss
        loss_function = loss(Y, y_pred)
        # gradients
        dw = gradient(X, Y, y_pred)
        # weight update
        w -= learning_rate * dw
        # print network summary
        if epoch % 2 == 0:
            print("loss: {loss_value}, epoch: {nth_epoch} weight: {nth_weight}"
                  .format(loss_value=loss_function, nth_epoch=epoch + 1, nth_weight=w))


if __name__ == '__main__':
    main()
