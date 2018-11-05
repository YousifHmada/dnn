import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return z * (z > 0)

def dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def dtanh(z):
    return 1 + (tanh(z) * tanh(z))

def drelu(z):
    return (z > 0)


#define hyper parameters
layers = [
    (2, 'relu'), (5, 'relu'), (4, 'relu'), (5, 'relu'), (1, 'tanh')
]
L = len(layers) - 1
m = 200
learning_rate = 0.01
num_iteration = 10000

#define Inputs & outputs
n_x = layers[0][0]
n_y = layers[L][0]
X = np.random.rand(n_x, m)
Y = np.random.rand(n_y, m)

#initialize parameters
W = [None]*(L+1)
b = [None]*(L+1)
for l in range(1, L + 1):
    W[l] = np.random.randn(layers[l][0], layers[l-1][0]) * 0.01
    b[l] = np.zeros((layers[l][0], 1))

Z = [None]*(L+1)
A = [None]*(L+1)
A[0] = X
dA = [None]*(L+1)
dZ = [None]*(L+1)
dW = [None]*(L+1)
db = [None]*(L+1)
cost = None

for iteration in range(num_iteration):
    
    #forward propagation
    for l in range(1, L + 1):
        Z[l] = np.dot(W[l], A[l - 1]) + b[l]
        if layers[l][1] == 'sigmoid':
            A[l] = sigmoid(Z[l])
        elif layers[l][1] == 'tanh':
            A[l] = tanh(Z[l]) 
        elif layers[l][1] == 'relu':
            A[l] = relu(Z[l])

    #compute cost
    cost = (-1/m) * np.sum(Y * np.log(A[L])+(1 - Y) * np.log(1 - A[L]))
    print(cost)

    #bacward propagation
    dA[L] = -(Y / A[L]) - ((1 - Y) / (1 - A[L]))
    for l in reversed(range(1, L + 1)):
        if layers[l][1] == 'sigmoid':
            dZ[l] = dA[l] * dsigmoid(Z[l])
        elif layers[l][1] == 'tanh':
            dZ[l] = dA[l] * dtanh(Z[l])
        elif layers[l][1] == 'relu':
            dZ[l] = dA[l] * drelu(Z[l])
        dW[l] = (1/m) * np.dot(dZ[l], A[l - 1].T)
        db[l] = (1/m) * np.sum(dZ[l], axis=1, keepdims=True)
        dA[l - 1] = np.dot(W[l].T, dZ[l])

    #update parameters
    for l in range(1, L + 1):
        W[l] += learning_rate * dW[l]
        b[l] += learning_rate * db[l]

print("cost : " + str(cost))
print("Y : ", Y)
print("Y^ : ", A[L])