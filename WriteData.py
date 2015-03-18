__author__ = 'Vishal'
#imports
import math

if __name__ == '__main__':
    f = open('data.txt', 'w')
    taylor = 3
    for x in xrange(1, 10):
        y = x*1.0/2
        #y = x*.1
        print y
        tString = str(y)
        for t in xrange(2, taylor+1):
            tString = tString + ", " + str(math.pow(y, t))
        f.write(tString + " | " + str(y) + ", " + str(y*y) +"\n")
    #for x in xrange(0, 19):
       # f.write(str(math.radians(x*10)) + ", " + str(math.pow(math.radians(x*10), 2)) + ", " + str(math.pow(math.radians(x*10), 3)) + " | " + str(math.sin(math.radians(x*10))) + "\n")
    f.close()
