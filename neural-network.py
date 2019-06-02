import numpy as np
import random
import matplotlib.pyplot as plt
from functools import reduce

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
        neurons = []
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
            neurons.append(a)
        
        return neurons
    
    def SGB(self, training_data, epochs, eta):
        training_data = list(training_data)
        correct_data = training_data[1]
        input_data = training_data[0]
        n = len(input_data)
        
        if(n != len(correct_data)):
            print("Array malformatado")
            return
        
        
        erros = []
        for count in range(0, epochs):
            aux_error = []
            for inputs, correct in zip(input_data, correct_data):
                self.update_mini_batch(inputs, correct, aux_error, eta)
            
            erros.append(reduce(lambda x,y : [np.sum(w+z/len(input_data)) for w,z in zip(x,y)], aux_error))
        
        self.plotResults(epochs, erros)
    
    def plotResults(self, epochs, erros):
        fig, ax = plt.subplots()  
        ax.plot(np.arange(epochs), erros, 'r')  
        ax.set_xlabel('Iterações')  
        ax.set_ylabel('Custo')  
        ax.set_title('MSE vs. Epoch')
        plt.show()
    
    def update_mini_batch(self, inputs, correct, error, eta):
        nabla_w, nabla_b = self.backprop(inputs, correct, error)
        new_weights = []
        new_biases = []
        
        for delta_w, delta_b, w, b in zip(nabla_w[::-1], nabla_b[::-1], self.weights, self.biases):
            new_weights.append(w-delta_w.T*eta)
            new_biases.append(b-delta_b*eta) #Removi a transposta talvez de merda
        self.__biases = new_biases
        self.__weights = new_weights
    
    def backprop(self, inputs, correct, error):
        inputs = np.array([inputs]).T
        correct = np.array([correct]).T
        
        # print("correct ", correct)
        neurons = self.feedforward(inputs)
        result = neurons[-1]
        
        neurons.pop(-1)
        neurons.insert(0,inputs)
        
        nabla_w = []
        nabla_b = []
        
        error.append(self.cost(result, correct))
        aux = self.cost_derivation(result, correct) * self.function_prime(result)      
        
        for n, w, b in zip(neurons[::-1], self.weights[::-1], self.__biases[::-1]):
            # Atualiza os bias
            nabla_b.append(aux)
            aux = np.dot(n, aux.T)
            
            # Atualiza os pesos
            nabla_w.append(aux)
            
            aux = aux * w.T
            aux = np.array([aux.sum(axis=1)]).T
            aux = aux * self.function_prime(n)
        return nabla_w, nabla_b
    
    # FUNÇÃO DE CUSTO A SER UTILIZADA
    def cost(self, result, correct):
        difference = result - correct
        return list(map(lambda x: np.square(x) , difference))
    
    # DERIVADA DA FUNÇÃO DE CUSTO
    def cost_derivation(self, result, correct):
        return result-correct

    # DERIVADA DA FUNÇÃO DE ATIVAÇÃO
    def function_prime(self, g_x):
        return g_x * (1 - g_x)
        

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


net = Network([4,10,2])
net.SGB([[[0,0,0,0],[0,1,0,0], [1,0,0,0], [1,1,0,0]], [[1,1], [0,0], [0,0], [1,1]]], 5000, 0.5)
# net.SGB([[[0],[1]], [[1,1], [0,0]]], 5000, 0.5)