__author__ = 'Vishal'
#imports
import random
import math

#classes


class Neuron:

    def __init__(self):
        self.inputs = []
        self.weights = []
        self.biasLoc = 0
        self.output = 0

    def addInput(self, put):
        self.inputs.append(put)
        self.weights.append(Neuron.chooseRandom())

    def addLayerInputs(self, nl):
        assert isinstance(nl, NeuronLayer)
        for x in nl.neurons:
            self.addInput(x.output)

    def makeBias(self, x):
        self.biasLoc = len(self.inputs)
        self.inputs.append(1)
        self.weights.append(x)

    def calcOutput(self):
        someOutput = 0
        for x in xrange(len(self.inputs)):
            blah = someOutput
            someOutput = someOutput + self.inputs[x]*self.weights[x]

        self.output = Neuron.sigmoid(someOutput)
        #return self.output

    @staticmethod
    def sigmoid(x):
        #return 1 / (1 + math.pow(math.e, -1.0*x))
        return (math.tanh(x)+1)*.5

    @staticmethod
    def chooseRandom():
        amp = 1
        return ((random.random()*amp*2)-amp)

class NeuronLayer:

    def __init__(self):
        self.neurons = []

    def addNeuron(self, n):
        assert isinstance(n, Neuron)
        self.neurons.append(n)


class NeuronNet:

    def __init__(self):
        self.layers = []
        self.value = []

    def addLayer(self, l):
        assert isinstance(l, NeuronLayer)
        self.layers.append(l)

    def update(self, values):
        if len(self.layers[0].neurons) != len(values):
            return 0
        for x in xrange(len(self.layers[0].neurons)):
            self.layers[0].neurons[x].inputs[0] = values[x]
            self.layers[0].neurons[x].output = values[x]
        for x in xrange(1, len(self.layers)):
            for z in xrange(len(self.layers[x].neurons)):
                for y in xrange(len(self.layers[x-1].neurons)):
                    self.layers[x].neurons[z].inputs[y] = self.layers[x-1].neurons[y].output
                self.layers[x].neurons[z].calcOutput()

        self.value = []
        for x in xrange(len(self.layers[-1].neurons)):
            self.value.append(self.layers[-1].neurons[x].output)
        #self.layers[-1].neurons[0].output

        return self.value