import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predict = torch.sigmoid(self.linear(x))
        return y_predict


def main():
    # Load dataset
    breast_cancer = datasets.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target

    n_samples, n_features = X.shape
    print(f"n_samples: {n_samples}, n_features: {n_features}")

    [X_train, X_test, y_train, y_test] = train_test_split(X, y,
                                                          test_size=0.2,
                                                          random_state=42)
    # Scale data
    standard_scalar = StandardScaler()
    X_train = standard_scalar.fit_transform(X_train)
    X_test = standard_scalar.transform(X_test)

    # Convert from numpy's array to torch's tensor
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    y_train = y_train.view(y_train.shape[0], 1)
    y_test = y_test.view(y_test.shape[0], 1)

    # Parameters
    learning_rate = 0.1
    n_epochs = 100

    model = LogisticRegression(n_features)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        # forward pass
        y_predicted = model(X_train)
        loss = criterion(y_predicted, y_train)
        # backward pass
        loss.backward()
        # update
        optimizer.step()
        # zero gradients
        optimizer.zero_grad()

        if (epoch+1) % 10 == 0:
            print("loss: {loss_value:.3f}, "
                  "epoch: {nth_epoch}"
                  .format(loss_value=loss, nth_epoch=epoch + 1))

    with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted_class = y_predicted.round()
        acc = y_predicted_class.eq(y_test).sum() / float(y_test.shape[0])
        print(f"test accuracy = {acc:.4f}")


if __name__ == '__main__':
    main()
