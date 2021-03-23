import numpy as np

# create arrays f = 3x
X = np.array([1, 2, 3, 4])
Y = np.array([3, 6, 9, 12])
# weight initialization
w = np.array([0.1, 0.2, 0.13, 0.4])

# forward propagation step
def forward(x):
    return x * w

# define loss function
def loss(y, y_pred):
    return np.mean(np.square(y - y_pred))




def main():





if __name__ == '__main__':
    main()