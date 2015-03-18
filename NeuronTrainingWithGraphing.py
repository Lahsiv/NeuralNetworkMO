__author__ = 'Vishal'

#imports
import random
import math
import NeuronNetwork
from NeuronNetwork import NeuronNet
from NeuronNetwork import NeuronLayer
from NeuronNetwork import Neuron
import matplotlib.pyplot as plt
import time

#global variables
startingNumOfNeurons = 6 #in the lowest non-input layer
decayOfNeurons = 5#how many less neurons each successive layer has
numTrials = 50000 #number of iterations of the error
displayProgress = 500 #number of iterations for the computer to display progress
learningRate = 0.01 #rate at which the algorithm learns

textXLoc = .68
textYLoc = .75

picNum = 1

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
    while(n > 0):
        nl = NeuronLayer()
        for x in xrange(n):
            neur = Neuron()
            neur.addLayerInputs(nn.layers[-1])
            neur.makeBias(random.random())
            neur.calcOutput()
            nl.addNeuron(neur)
        nn.addLayer(nl)
        n -= decay
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
            print "\tWeight: " + str(n.weights[0])
            print ""

#graphing methods
def updateData(h, newxdata, newydata):
    h.set_xdata(newxdata)
    h.set_ydata(newydata)
    plt.draw()
    #plt.savefig("out" + str(picNum) + ".png")


#main
if __name__ == '__main__':
    nNet = NeuronNet()
    lines = open('data.txt', 'r').read().replace(" ", "").split("\n")
    begins = []
    ends = []
    for x in lines:
        y = x.split("|")
        if y == [""]: continue
        begins.append(y[0].split(","))
        for z in xrange(len(begins[-1])):
            begins[-1][z] = float(begins[-1][z])
        ends.append(float(y[1]))
    #start scaling output
    endsCopy = ends[:]
    minOutput = min(ends)
    maxOutput = max(ends)
    outputRange = maxOutput - minOutput
    for counter in xrange(len(ends)):
        ends[counter] = (ends[counter] - minOutput) / outputRange
    #done scaling output
    #done initializing input
    nNet = init(nNet, startingNumOfNeurons, decayOfNeurons, begins[0])
    #start initializing graphing
    errorText = plt.figtext(textXLoc, textYLoc, "SSE:")
    xValues = []
    for x in begins:
        xValues.append(x[0])
    #baseX = numpy.arange(begins[0][0], begins[-1][0], .02)
    #y = numpy.sin(baseX)
    plt.ion()
    plt.plot(xValues, endsCopy)
    plt.show()
    plotData, = plt.plot([], [])

    #initialized the neural net with random weights
    trialCount = 1
    #start training the neural net
    #algorithm found at http://www.cheshireeng.com/Neuralyst/nnbg.htm
    while trialCount < numTrials:
        cValues = []
        for cInput in xrange(len(begins)):
            preWeights = []
            realOutput = nNet.update(begins[cInput])
            eTerm = realOutput*(1-realOutput)*(ends[cInput] - realOutput)
            preWeights.append([])
            preWeights[0].append(eTerm)
            x = 0
            lim = len(nNet.layers[-1].neurons[0].weights) - 1
            preWeights[0].append([])
            while x < lim:
                preWeights[0][1].append(nNet.layers[-1].neurons[0].weights[x])
                nNet.layers[-1].neurons[0].weights[x] += learningRate * eTerm * nNet.layers[-1].neurons[0].inputs[x]
                x += 1
            #done adjusting weights for output layer
            #start backpropagating for weights in internal layers
            cLayer = len(nNet.layers) - 2
            while cLayer >= 0:
                #loops through neurons in this layer
                for cN in xrange(len(nNet.layers[cLayer].neurons)):
                    totalErr = 0
                    for x in xrange(len(nNet.layers[cLayer+1].neurons)):
                        totalErr += preWeights[x][0]*preWeights[x][1][cN]
                    #done calculating the sum of the earlier errors for current neuron
                    eTerm = totalErr*nNet.layers[cLayer].neurons[cN].output*(1-nNet.layers[cLayer].neurons[cN].output)
                    preWeights.append([])
                    preWeights[-1].append(eTerm)
                    preWeights[-1].append([])
                    for sInput in xrange(len(nNet.layers[cLayer].neurons[cN].inputs)-1):
                        preWeights[-1][1].append(nNet.layers[cLayer].neurons[cN].weights[sInput])
                        nNet.layers[cLayer].neurons[cN].weights[sInput] += learningRate*eTerm*nNet.layers[cLayer].neurons[cN].inputs[sInput]
                for someCount in xrange(len(nNet.layers[cLayer+1].neurons)):
                    preWeights.pop(0)
                cLayer -= 1
            cValues.append(nNet.value)
            #done looping through layers
        if trialCount % displayProgress == 0:
            print "Done with " + str(trialCount) + " trials"
            tError = 0
            for counter in xrange(len(cValues)):
                cValues[counter] = cValues[counter]*outputRange + minOutput
            for someValue in xrange(len(cValues)):
                tError += math.pow(math.fabs(endsCopy[someValue] - cValues[someValue]), 2)
                errorText.set_text("SSE: " + str(tError) + "\n\n" + str(int(trialCount*100/numTrials)) + "% done")
            updateData(plotData, xValues, cValues)
            picNum += 1
        trialCount += 1
    errorText.set_text("SSE: " + str(tError) + "\n\nDone training network!")
    plt.draw()
    #done setting weights
    print "\n\nDone creating chromosome. Sample results below.\n\n"
    for x in xrange(5):
        v = math.radians(random.random()*180)
        testArr = []
        for x in xrange(len(begins[0])):
            testArr.append(math.pow(v, x+1))
        print testArr
        nNet.update(testArr)
        print "V: " + str(v)
        print "Sin(V): " + str(math.sin(v))
        print "Estimated: " + str(nNet.value)
        print ""
    time.sleep(8)

