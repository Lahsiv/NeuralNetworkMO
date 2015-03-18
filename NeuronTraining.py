__author__ = 'Vishal'

#imports
import random
import math
import NeuronNetwork
from NeuronNetwork import NeuronNet
from NeuronNetwork import NeuronLayer
from NeuronNetwork import Neuron
import time
import pickle
from sys import stdout as wr

#global variables
startingNumOfNeurons = 5 #in the lowest non-input layer
decayOfNeurons = 3#how many less neurons each successive layer has

neurArray = [3, 2] #array that holds the neuron pattern

numTrials = 5000 #number of iterations of the error
displayProgress = 20 #number of iterations for the computer to display progress
learningRate = 0.1 #rate at which the algorithm learns

textXLoc = .68
textYLoc = .75

def init(nn, startNeurons, decay, startInput):
    """
    @rtype : NeuronNet
    """
    assert isinstance(nn, NeuronNet)
    nl = NeuronLayer()
    for l in startInput:
        neur = Neuron()
        neur.addInput(l)
        neur.weights[-1] = 1
        neur.output = l
        nl.addNeuron(neur)
    nn.addLayer(nl)

    n = startNeurons
    for n in neurArray:
        nl = NeuronLayer()
        for x in xrange(n):
            neur = Neuron()
            neur.addLayerInputs(nn.layers[-1])
            neur.makeBias(random.random())
            neur.calcOutput()
            nl.addNeuron(neur)
	nn.addLayer(nl)
    '''while n > 0:
        nl = NeuronLayer()
        for x in xrange(n):
            neur = Neuron()
            neur.addLayerInputs(nn.layers[-1])
            neur.makeBias(random.random())
            neur.calcOutput()
            nl.addNeuron(neur)
        nn.addLayer(nl)
        n -= decay'''
    #for cNeuron in xrange(len(nn.layers[-1].neurons)):
        #nn.layers[-1].neurons[cNeuron].skipSigmoid = True
    return nn

def display(nn, onlyOutput):
    assert isinstance(nn, NeuronNet)
    for iu in xrange(len(nn.layers)):
        print "Layer " + str((iu+1))
        for n in nn.layers[iu].neurons:
            if(not onlyOutput):
                print "\tInputs:" + str(n.inputs)
                print "\tWeights:" + str(n.weights)
            print "\tOutput: " + str(n.output)
            print ""

def displayBias(nn):
    assert isinstance(nn, NeuronNet)
    for iu in xrange(len(nn.layers)):
        print "Layer " + str((iu+1))
        for n in nn.layers[iu].neurons:
            print "\tBias: " + str(n.inputs[-1])
            print "\tWeight: " + str(n.weights)
            print ""

#main
if __name__ == '__main__':
    nNet = NeuronNet()
    lines = open('data.txt', 'r').read().replace(" ", "").split("\n")
    begins = []
    ends = []
    for x in lines:
        y = x.split("|")
        if y == [""]:
            continue
        begins.append(y[0].split(","))
        for z in xrange(len(begins[-1])):
            begins[-1][z] = float(begins[-1][z])
        ends.append(y[1].split(","))
        for z in xrange(len(ends[-1])):
            ends[-1][z] = float(ends[-1][z])

    #done initializing input
    nNet = init(nNet, startingNumOfNeurons, decayOfNeurons, begins[0])

    #initialized the neural net with random weights
    trialCount = 1
    #start training the neural net
    #algorithm found at http://www.cheshireeng.com/Neuralyst/nnbg.htm
    while trialCount < numTrials:
        cValues = []
        for cInput in xrange(len(begins)):
            preWeights = []
            #preWeights.append([])
            realOutput = nNet.update(begins[cInput])
            for cFOutput in xrange(len(realOutput)):
                preWeights.append([])
                eTerm = realOutput[cFOutput]*(1-realOutput[cFOutput])*(ends[cInput][cFOutput] - realOutput[cFOutput])
                preWeights[-1].append(eTerm)
                x = 0
                lim = len(nNet.layers[-1].neurons[cFOutput].weights) 
                preWeights[-1].append([])
                while x < lim:
                    preWeights[-1][1].append(nNet.layers[-1].neurons[cFOutput].weights[x])
                    nNet.layers[-1].neurons[cFOutput].weights[x] += learningRate * eTerm * nNet.layers[-1].neurons[cFOutput].inputs[x]
                    x += 1
            #done adjusting weights for output layer
            #start backpropagating for weights in internal layers
            cLayer = len(nNet.layers) - 2
            while cLayer >= 0:
                #loops through neurons in this layer
                for cN in xrange(len(nNet.layers[cLayer].neurons)):
                    totalErr = 0
                    '''print "---"
                    print preWeights[0][0]
                    print "+++"
                    print preWeights[0][1]'''
                    for x in xrange(len(nNet.layers[cLayer+1].neurons)):
                        totalErr += preWeights[x][0]*preWeights[x][1][cN]
                    #done calculating the sum of the earlier errors for current neuron
                    eTerm = totalErr*nNet.layers[cLayer].neurons[cN].output*(1-nNet.layers[cLayer].neurons[cN].output)
                    preWeights.append([])
                    preWeights[-1].append(eTerm)
                    preWeights[-1].append([])
                    for sInput in xrange(len(nNet.layers[cLayer].neurons[cN].inputs)):###-1
                        preWeights[-1][1].append(nNet.layers[cLayer].neurons[cN].weights[sInput])
                        nNet.layers[cLayer].neurons[cN].weights[sInput] += learningRate*eTerm*nNet.layers[cLayer].neurons[cN].inputs[sInput]
                for someCount in xrange(len(nNet.layers[cLayer+1].neurons)):
                    preWeights.pop(0)
                cLayer -= 1
            cValues.append(nNet.value)
            #done looping through layers
        if trialCount % displayProgress == 0:
            wr.write("\r" + "Done with " + str(trialCount) + " trials")
        wr.flush()
        trialCount += 1
    #done setting weights
    print "\n\nDone creating network. Sample results below.\n\n"

    print "\n\n\n"
    print begins
    for x in xrange(len(begins)):
        nNet.update(begins[x])
        print nNet.value
        print ends[x]
        print "---"

    sN = open("savednetwork.txt", "w")
    pickle.dump(nNet, sN)
    sN.close()
    displayBias(nNet)
