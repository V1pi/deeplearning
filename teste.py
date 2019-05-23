import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class Teste(object):
    def __init__(self, sizes):
        self.__num_layers = len(sizes)
        self.__sizes = sizes
        self.__biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.__weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            print("Biase: ", b)
            print("weights: ", w)
            a = sigmoid(np.dot(w, a)+b)
        
        print("RESULTADO ", a)
        return a

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
    
sizes = [1,2,3]
sizes2 = [1,2,3,4]
#print("[:-1]", sizes[:-1])
#print("[1:]", sizes[1:])
#print("zip", [np.random.randn(y, 1) for y in sizes[1:]])

teste = Teste(sizes)
teste.feedforward([[1]])
print(teste.biases)
print(teste.weights)