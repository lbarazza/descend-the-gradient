import numpy as np

points = np.genfromtxt("data.csv", delimiter=',')

def error(points, m, b):
    error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        error += (y - (m * x + b))**2
    error /= float(len(points))
    return error

def train(points, m, b, learning_rate, training_epochs):
    for _ in range(0, training_epochs):
        delta_m = 0
        delta_b = 0
        for i in range(len(points)):
            x = points[i, 0]
            y = points[i, 1]
            delta_m -= x * (y - (m * x + b))
            delta_b -= y - (m * x + b)
        delta_m *= 2/len(points)
        delta_b *= 2/len(points)
        m -= learning_rate * delta_m
        b -= learning_rate * delta_b
    return(m, b)

training_epochs = 1000
learning_rate = 0.0001

m = 0
b = 0

print("m: ", m)
print("b: ", b)
print("Error: ", error(points, m, b))

m, b = train(points, m, b, learning_rate, training_epochs)

print("New m: ", m)
print("New b: ", b)
print("Error: ", error(points, m, b))
