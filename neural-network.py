import numpy as np
import random

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self, sizes):
        self.__num_layers = len(sizes)
        self.__sizes = sizes
        self.__biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.__weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        idx = 0
        print("LISTA a: ", list(a))
        for aux in list(a):
            aux = np.transpose([aux])
            print("aux: ", aux)
            for b,w in zip(self.biases, self.weights):
                aux = sigmoid(np.dot(w, aux)+b)
            a[idx] = aux
            idx+=1
        return a
    
    def SGB(self, training_data, epochs, eta):
        training_data = list(training_data)
        n = len(training_data)

        a = self.feedforward(training_data[0])
        print("result: ", a)

    @property
    def sizes(self):
        return self.__sizes
    @property
    def biases(self):
        return self.__biases
    @property
    def num_layers(self):
        return self.__num_layers
    @property
    def weights(self):
        return self.__weights

net = Network([2,2,3])
net.SGB([[[0,1],[1,1]], [[1],[0]]], 200, 0.5)