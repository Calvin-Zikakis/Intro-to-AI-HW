import random
import inspect
import sys

def weightedChoice(elements, weights):
    # verify the lists are the same length
    assert len(elements) == len(weights)
    total = sum(weights)
    r = random.uniform(0, total)
    w = 0
    for i in range(len(elements)):
        w += weights[i]
        if w > r:
            return elements[i]
    # all weights are zero if we get here, so pick at random
    return random.choice(elements)

def Average(lst):
    return sum(lst) / len(lst)

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)