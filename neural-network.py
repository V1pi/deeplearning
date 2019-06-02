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
        # idx = 0
        # print("LISTA a: ", list(a))
        # for aux in list(a):
        #     aux = np.transpose([aux])
        #     print("aux: ", aux)
        #     for b,w in zip(self.biases, self.weights):
        #         aux = sigmoid(np.dot(w, aux)+b)
        #     a[idx] = aux
        #     idx+=1
        neurons = []
        # print("weights: ", self.weights)
        # print("bias: ", self.biases)
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
                inputs = np.array([inputs]).T
                correct = np.array([correct]).T
                
                # print("correct ", correct)
                neurons = self.feedforward(inputs)
                result = neurons[-1]
                
                neurons.pop(-1)
                neurons.insert(0,inputs)
                
                nabla_w = []
                nabla_b = []
                
                # print("neurons", neurons)
                # print("result", result)
                error = result - correct
                # aux_error += error[0][0]*error[0][0]
                aux_error.append(list(map(lambda x: np.square(x) , error)))
                aux = error * result * (1-result)
                # print("aux_error ", aux_error)
                # print("error ", error)
                
                
                
                for n, w, b in zip(neurons[::-1], self.weights[::-1], self.__biases[::-1]):
                    # Atualiza os bias
                    # print("bias ", aux)
                    nabla_b.append(aux)
                    # print("neurons_layer ", n)
                    # print("neurons ", n)
                    aux = np.dot(n, aux.T)
                    
                    # Atualiza os pesos
                    nabla_w.append(aux)
                    
                    # print("pesos ", aux)
                    aux = aux * w.T
                    aux = np.array([aux.sum(axis=1)]).T
                    # print("Soma dos pesos ", aux)
                    aux = aux * n * (1 - n)
                    
                    # print("aux ", aux)
                    # aux = aux * n * (1-n)
                    
                    # aux = aux * w.transpose()
                    
                    # aux = np.array([aux.sum(axis=1)])
                    # print("n ", n)
                    # print("w ", w)
                    # print("b ", b)
                # print("Nabla_b", nabla_b)
                # print("Nabla_w", nabla_w)
                new_weights = []
                new_biases = []
                for delta_w, delta_b, w, b in zip(nabla_w[::-1], nabla_b[::-1], self.weights, self.biases):
                    # print("bias ", b)
                    # print("delta_bias ", delta_b)
                    # print("weights ", w)
                    # print("delta_weights ", delta_w)
                    new_weights.append(w-delta_w.T*eta)
                    new_biases.append(b-delta_b*eta) #Removi a transposta talvez de merda
                # print("old_bias ", self.biases)
                # print("new_boas ", new_biases)
                self.__biases = new_biases
                self.__weights = new_weights
                # break
            
            # print("Soma ", reduce(lambda x,y : [np.sum(w+z/len(input_data)) for w,z in zip(x,y)], aux_error))
            erros.append(reduce(lambda x,y : [np.sum(w+z/len(input_data)) for w,z in zip(x,y)], aux_error))
            # print("erros ", erros)
            # erros.append(aux_error/len(input_data))
            # break
        print("Erros: ", 1)
        fig, ax = plt.subplots()  
        ax.plot(np.arange(epochs), erros, 'r')  
        ax.set_xlabel('Iterações')  
        ax.set_ylabel('Custo')  
        ax.set_title('MSE vs. Epoch')
        plt.show()
        
            
        

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


net = Network([4,1,1])
net.SGB([[[0,0,0,0],[0,1,0,0], [1,0,0,0], [1,1,0,0]], [[1], [0], [0], [1]]], 5000, 0.5)
# net.SGB([[[0],[1]], [[1,1], [0,0]]], 5000, 0.5)