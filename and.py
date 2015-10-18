import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1+np.exp(-x))

"""
    Predicts the outcome for the input x and the weights w
    x_0 is 1 and w_0 is the bias
"""
def predict(x,w):
    return sigmoid(np.sum([w[p]*x[p] for p in range(len(x))]))

"""
    Determine the cost of the prediction (pred)
"""
def cost(real,pred):
    return np.sqrt(1.0/2*np.mean(np.power(real-pred,2)))

"""
    get the gradient for the inputs x,weights w and the outputs y
"""
def gradient(x,w,y):
    grads = np.empty((len(w)))
    for j in range(len(w)):
        grads[j] = np.mean(np.sum([x[i,j]*(predict(x[i],w)-y[i]) for i in range(len(x))]))
    return grads

"""
    get the new weights based on the old ones w, learning rate a and the gradients
"""
def getWeights(w,a,grads):
    return w-a*grads

"""
    initialize the weights w, inputs x and outputs y
    here the outputs are the outputs for x[0] or x[1]
"""
w = np.array([0,0,0])
# the first value in each subarray is the weight of the bias input, which is always set to 1.
x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([0,0,0,1])

"""
    Use several steps to minimize the costs
"""
steps = 100
arr = np.empty((steps,2))
# the learning rate
a = 1
for c in range(steps):
    # Create a new array (prediction) which afterwards holds the prediction for the inputs x.
    prediction = np.empty((len(x)))
    for i in range(len(x)):
        # calculate the prediction for each input (keep the weights unchanged)
        prediction[i] = predict(x[i],w)
        print("prediction for: ",x[i]," is ",prediction[i])

    # get the costs of the prediction
    print("Cost: ")
    vCost = cost(y,prediction)
    print(vCost)
    # calculate the gradients
    print("Grads: ")
    grads = gradient(x,w,y)
    print(grads)
    # get the new weights using a learning rate of 0.3
    print("New Weights: ")
    w = getWeights(w,a,grads)
    print(w)
    # only for plotting the costs
    arr[c] = [c,vCost]

plt.scatter(arr[:,0],arr[:,1])
plt.xlabel('steps')
plt.ylabel('cost')
plt.title('Cost graph')
plt.show()
