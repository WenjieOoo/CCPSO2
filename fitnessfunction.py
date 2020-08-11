import math
import random


def function1(xlist):
    f = 0
    for x in xlist:
        f += (x-1314)**2
    f += 24
    return f


# Shifted Rastrigin’s Function
def function2(xlist):
    f = 0
    for x in xlist:
        f += (x-1314) ** 2 - 10 * math.cos(2 * math.pi * (x-1314)) + 10
    f += 24
    return f


# FastFractal “DoubleDip” Function
def twist(y):
    return 4 * (y**4 - 2 * (y**3) + y**2)


def doubledip(x, c, s):
    if -0.5 < x < 0.5:
        return (-6144 * (x-c)**6 + 3088 * (x-c)**4 - 392 * (x-c)**2 + 1) * s
    else:
        return 0


def ran1():
    return random.random()


def ran2():
    return random.choice([0, 1, 2])


def fractal1D(x):
    y = 0
    for k in range(1, 4):
        for j in range(2**(k-1)-1):
            for i in range(ran2()-1):
                y += doubledip(x, ran1(), 1 / (2**(k-1) * (2-ran1())))
    return y


def function3(xlist):
    f = 0
    for i in range(len(xlist)):
        f = f + fractal1D(xlist[i] + twist(xlist[(i+1) % (len(xlist))]))
    return f


