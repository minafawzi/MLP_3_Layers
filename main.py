import math
import decimal
import numpy as np
import matplotlib.pyplot as plt


# function to implement is
def hw6_Q2(p):
    # y= 1 + math.sin((math.pi/4)*p)
    y = math.exp(-1 * abs(p)) * math.sin(math.pi * p)
    return y


# Activation function implementation
def sigmoid(x):
    x = np.float64(x)
    exp_out = np.exp(-np.clip(x, -250, 250))
    return 1. / (1. + exp_out)


def purelin(x):
    return x


# Activation function derivatives
def sigmoid_derv(x):
    x = np.float64(x)
    return (1 - x) * x


def purelin_derv(x):
    return 1


# Vectorize the function to use it on the full array
hw6_Q2_vectorized = np.vectorize(hw6_Q2)

# populate input vector
P = np.linspace(-2, 2, 100)
# print(P)

# populate output
Y = hw6_Q2_vectorized(P)
plt.plot(P, Y, label='Original Function')
# print(Y)

# Network Design
#                          _ _ _[N1]_ _ _
#                         /      .       \
#                  P_ _ _/       .        \_ _ _[N]
#                        \       .        /
#                         \_ _ _[Nn]_ _ _/
#


Hidden_Layer_Neurons = 5
Learning_rate = 0.1
epochs = 100

Optimization = "Batch" # Optimization can be "SGD" or "Batch"
Gradient1_W = [0]*len(P)
Gradient2_W = [0]*len(P)

Gradient1_b = [0]*len(P)
Gradient2_b = [0]*len(P)

errors = [0]*len(P)
mse = [0]*epochs

# Initialize the network weight and bias
W1 = np.random.uniform(-0.5, 0.5, Hidden_Layer_Neurons)
b1 = np.random.uniform(-0.5, 0.5, Hidden_Layer_Neurons)

W2 = np.random.uniform(-0.5, 0.5, Hidden_Layer_Neurons)
b2 = np.random.uniform(-0.5, 0.5)

if Optimization == 'SGD':
    for i in range(epochs):
        #print(W2)
        for j in range(len(P)):
            # *** forward pass ****

            n1 = np.dot(W1, P[j]) + b1  # calculate n of first layer
            a1 = sigmoid(n1)  # calculate output using sigmoid function

            n2 = np.dot(W2, a1) + b2
            a2 = purelin(n2)
            error = Y[j] - a2

            #get square error and store in errors array
            errors[j] = error*error
            # **** backward pass ****

            # Calculate the last layer senstivity
            s2 = -2 * purelin_derv(n2) * error

            # Calculate the hidden layet senstivity
            # 1- Calculate Fn1 Matrix

            Fn1 = np.full((Hidden_Layer_Neurons, Hidden_Layer_Neurons), sigmoid_derv(a1))
            # print(Fn1)
            Fn1 = np.diag(np.diag(Fn1))

            # 2- Calculate s1
            s1 = W2.T * s2
            s1 = np.dot(Fn1, s1)

            # Update the weights
            W2 = W2 - Learning_rate * (s2 * a1.T)
            b2 = b2 - Learning_rate * s2

            W1 = W1 - Learning_rate * (np.dot(s1, P[j].T))
            b1 = b1 - Learning_rate * s1

        # print(i)
        mse[i] = np.sum(errors)/len(P)
        print(" ** Epoch", str(i), ">>>>>>  MSE :", str(mse[i]))

elif Optimization =="Batch":
    for i in range(epochs):
        #print(W2)
        for j in range(len(P)):
            # *** forward pass ****
            n1 = np.dot(W1, P[j]) + b1  # calculate n of first layer
            a1 = sigmoid(n1)  # calculate output using sigmoid function

            n2 = np.dot(W2, a1) + b2
            a2 = purelin(n2)
            error = Y[j] - a2

            errors[j] = error * error
            # **** backward pass ****

            # Calculate the last layer sensitivity
            s2 = -2 * purelin_derv(n2) * error

            # Calculate the hidden layer sensitivity
            # 1- Calculate Fn1 Matrix

            Fn1 = np.full((Hidden_Layer_Neurons, Hidden_Layer_Neurons), sigmoid_derv(a1))
            Fn1 = np.diag(np.diag(Fn1))

            # 2- Calculate s1
            s1 = np.dot(W2.T, s2)
            s1 = np.dot(Fn1, s1)

            Gradient2_W[j] = np.dot(s2, a1.T)
            Gradient1_W[j] = np.dot(s1, P[j].T)

        mse[i] = np.sum(errors) / len(P)
        # Update the weights and the bias at the end of the Epoch
        print(Learning_rate * np.sum(Gradient2_W)/len(P))
        W2 = W2 - Learning_rate * np.sum(Gradient2_W)/len(P)
        b2 = b2 - Learning_rate * np.sum(Gradient2_b)/len(P)
        W1 = W1 - Learning_rate * np.sum(Gradient1_W)/len(P)
        b1 = b1 - Learning_rate * np.sum(Gradient1_b)/len(P)

        print(" ** Epoch", str(i), ">>>>>>  MSE :", str(mse[i]))


def predict(p):
    n1 = np.dot(W1, p) + b1
    a1 = sigmoid(n1)
    n2 = np.dot(W2, a1) + b2
    return purelin(n2)


# Vectorize the function to use it on the full array
predict_vectorized = np.vectorize(predict)

# populate output
Y2 = predict_vectorized(P)
plt.plot(P, Y2, linestyle="dashed", label="predicted", color='red')
plt.show()

x1 = range(0,epochs)
plt.plot(x1, mse , 'g', label='Training loss')
plt.title('Training loss')
#plt.xscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
