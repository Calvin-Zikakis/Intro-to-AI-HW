"""
Automated tester program for Simple Genetic Algorithm assignment

Written by Jim Marshall
Sarah Lawrence College, spring 2015
http://science.slc.edu/~jmarshall

1. Put this file in the same folder as your program file.

2. Put "import GAinspector" (no quotes) at the top of your program file.

3. After loading your file into Python, type GAinspector.test() at the
   Python prompt to test everything, or GAinspector.test(<function_name>)
   to test individual functions, where <function_name> is one of: randomGenome,
   makePopulation, fitness, evaluateFitness, crossover, mutate, or selectPair.
   For example:
   >>> GAinspector.test(randomGenome)

"""

import inspect, sys

def is_genome_string(s):
    if type(s) != str or len(s) == 0:
        return False
    for x in s:
        if x not in ["0", "1"]:
            return False
    return True

def approx_equal(x, y, tolerance=1e-10):
    return abs(x - y) < tolerance

def test_randomGenome(func, size):
    print("testing randomGenome(%d)" % (size), end=' ')
    if len(inspect.getargspec(func)[0]) != 1:
        raise Exception("wrong number of input parameters (should be 1)")
    try:
        result = func(size)
    except Exception as e:
        raise e
    if type(result) != str:
        raise Exception("didn't return a string")
    elif len(result) != size:
        raise Exception("returned a string of the wrong length")
    elif not is_genome_string(result):
        raise Exception("returned a non-binary string")
    # all tests passed
    print("... passed!")

def test_makePopulation(func, size, length):
    print("testing makePopulation(%d, %d)" % (size, length), end=' ')
    if len(inspect.getargspec(func)[0]) != 2:
        raise Exception("wrong number of input parameters (should be 2)")
    try:
        result = func(size, length)
    except Exception as e:
        raise e
    if type(result) != list:
        raise Exception("didn't return a list")
    elif len(result) != size:
        raise Exception("returned a list of the wrong length")
    else:
        # result list should contain only genome strings
        for x in result:
            if not is_genome_string(x):
                raise Exception("list contains an invalid genome: %s" % x)
            elif len(x) != length:
                raise Exception("list contains a genome of the wrong length")
    # all tests passed
    print("... passed!")
    
def test_fitness(func, genome, correctAnswer):
    print("testing fitness('%s')" % (genome), end=' ')
    if len(inspect.getargspec(func)[0]) != 1:
        raise Exception("wrong number of input parameters (should be 1)")
    try:
        result = func(genome)
    except Exception as e:
        raise e
    if type(result) != int:
        raise Exception("didn't return an integer")
    elif result != correctAnswer:
        raise Exception("returned %d instead of %d" % (result, correctAnswer))
    # all tests passed
    print("... passed!")

def test_evaluateFitness(func, population, correctAvg, correctBest, popnum=""):
    print("testing evaluateFitness(pop%s, %g, %g)" % (popnum, correctAvg, correctBest), end=' ')
    if len(inspect.getargspec(func)[0]) != 1:
        raise Exception("wrong number of input parameters (should be 1)")
    try:
        result = func(population)
    except Exception as e:
        raise e
    if type(result) not in [tuple,list] or len(result) != 2 or \
            type(result[0]) not in [float,int] or type(result[1]) not in [float,int]:
        raise Exception("didn't return a pair of numbers")
    if type(result[0]) != float:
        raise Exception("didn't return average fitness as a float")
    avg, best = result
    if not approx_equal(avg, correctAvg):
        raise Exception("returned incorrect average fitness")
    if not approx_equal(best, correctBest):
        raise Exception("returned incorrect best fitness")
    # all tests passed
    print("... passed!")

def test_crossover(func, gen1, gen2):
    print("testing crossover('%s', '%s')" % (gen1, gen2), end=' ')
    if len(inspect.getargspec(func)[0]) != 2:
        raise Exception("wrong number of input parameters (should be 2)")
    try:
        result = func(gen1, gen2)
    except Exception as e:
        raise e
    if type(result) not in [tuple,list] or len(result) != 2 or type(result[0]) != str or type(result[1]) != str:
        raise Exception("didn't return a pair of strings")
    off1, off2 = result
    if len(off1) != len(off2) or len(off1) != len(gen1):
        raise Exception("returned strings of the wrong length")
    # see if we get the expected behavior
    for i in range(len(gen1)*50):
        off1, off2 = func(gen1, gen2)
        success = False
        for i in range(1, len(gen1)):
            if off1 == gen1[:i] + gen2[i:] and off2 == gen2[:i] + gen1[i:] or \
               off1 == gen2[:i] + gen1[i:] and off2 == gen1[:i] + gen2[i:]:
                success = True
        if not success:
            raise Exception("crossover operation did not work")
    # all tests passed
    print("... passed!")

def test_mutate(func, genome, rate):
    print("testing mutate('%s', %g)" % (genome, rate), end=' ')
    if len(inspect.getargspec(func)[0]) != 2:
        raise Exception("wrong number of input parameters (should be 2)")
    try:
        result = func(genome, rate)
    except Exception as e:
        raise e
    if type(result) != str:
        raise Exception("didn't return a string")
    elif len(result) != len(genome):
        raise Exception("returned a string of the wrong length")
    elif not is_genome_string(result):
        raise Exception("returned an invalid genome string")
    # see if we get the expected behavior
    trials = 30000
    num_flipped = 0
    for i in range(trials):
        g = func(genome, rate)
        num_flipped += sum([1 if g[i] != genome[i] else 0 for i in range(len(genome))])
    num_bits = len(genome)*trials
    percent = 100.0*num_flipped/num_bits
    expected_percent = 100.0*rate
    if not approx_equal(percent, expected_percent, 0.20) or \
            rate == 0 and num_flipped != 0 or \
            rate == 1 and num_flipped != len(genome)*trials:
        print("\n%d expected bit flips (%.2f%% of bits)\n%d actual bit flips (%.2f%% of bits)" % \
            (num_bits*rate, expected_percent, num_flipped, percent))
        raise Exception("percent of bits flipped isn't right")
    # all tests passed
    print("... passed!")

def test_selectPair(func, population, popnum=""):
    print("testing selectPair(pop%s)" % popnum, end=' ')
    if len(inspect.getargspec(func)[0]) != 1:
        raise Exception("wrong number of input parameters (should be 1)")
    try:
        result = func(population)
    except Exception as e:
        raise e
    if type(result) not in [tuple,list] or len(result) != 2 or type(result[0]) != str or type(result[1]) != str:
        raise Exception("didn't return a pair of strings")
    if result[0] not in population or result[1] not in population:
        raise Exception("returned a string that is not in the population")
    # see if we get the expected behavior
    weights = [g.count('1') for g in sorted(population)]
    expected_distribution = [100.0*w/sum(weights) for w in weights]
    stats = {}
    for g in population:
        stats[g] = 0
    for i in range(10000):
        g1, g2 = func(population)
        if g1 not in population or g2 not in population:
            raise Exception("returned a string that is not in the population")
        stats[g1] += 1
        stats[g2] += 1
    total = sum(stats.values())
    actual_distribution = [100.0*stats[g]/total for g in sorted(stats)]
    for (expected, actual) in zip(expected_distribution, actual_distribution):
        if not approx_equal(actual, expected, 1.1):
            print()
            print_distribution("expected", expected_distribution)
            print_distribution("actual", actual_distribution)
            raise Exception("distribution of selected genomes isn't right")
    # all tests passed
    print("... passed!")

def print_distribution(type, dist):
    for x in dist:
        print("%.2f" % x, end=' ')
    print(" (%s)" % type)

def get_matching_name(name):
    main = sys.modules["__main__"]
    for x in dir(main):
        if x.lower() == name.lower():
            return x
    return None

def test(func=None):
    funcNames = ["randomGenome", "makePopulation", "fitness", "evaluateFitness",
                 "crossover", "mutate", "selectPair"]
    if func == None:
        main = sys.modules["__main__"]
        for name in funcNames:
            if name in dir(main):
                func = main.__getattribute__(name)
                test(func)
            elif name.lower() in [x.lower() for x in dir(main)]:
                # function is miscapitalized
                print("ERROR: the function %s must be named \"%s\"" % (get_matching_name(name), name))
            else:
                print("WARNING: %s is not defined ... skipping" % name)
        return
    if type(func) == str:
        print("ERROR: the function name should not be in quotes")
        return
    if not inspect.isfunction(func):
        print("ERROR: %s is not a valid function to test" % func)
        return
    name = func.__name__
    if name not in funcNames:
        # maybe it's just miscapitalized
        for fname in funcNames:
            if name.lower() == fname.lower():
                print("ERROR: this function must be named \"%s\"" % fname)
                return
    try:
        # the values for the individual test cases can be changed as desired,
        # but the ones below provide a reasonably good "workout", and will
        # catch most errors
        if name == "randomGenome":
            test_randomGenome(func, 20)
            test_randomGenome(func, 50)
        elif name == "makePopulation":
            test_makePopulation(func, 10, 30)
            test_makePopulation(func, 20, 40)
        elif name == "fitness":
            test_fitness(func, "111111", 6)
            test_fitness(func, "0000000", 0)
            test_fitness(func, "0000100000", 1)
            test_fitness(func, "100110101111010", 9)
        elif name == "evaluateFitness":
            pop1 = ["1000001110", "0101100101", "1000101101", "0001011011", "0101000001"]
            test_evaluateFitness(func, pop1, 4.4, 5, 1)
            pop2 = ["1111101100", "0000000100", "1000110010", "0010011011", "1001110000"]
            test_evaluateFitness(func, pop2, 4.2, 7, 2)
            pop3 = ["1011011111", "0001011010", "1010110001", "1001110100", "1100001100"]
            test_evaluateFitness(func, pop3, 5.2, 8, 3)
            pop4 = ["0011111011000", "1001101110101", "0000110100110", "0010011111101", "1111011111001",
                    "0001000011110", "0111000110010", "1110011100001", "1011110010010", "0010111100111",
                    "0001110110111", "1100111011100", "0001000101000", "1001111001010", "0100101111100"]
            test_evaluateFitness(func, pop4, 6.93333333333, 10, 4)
        elif name == "crossover":
            test_crossover(func, "010101", "100111")
            test_crossover(func, "0000000", "0000000")
            test_crossover(func, "1111111", "0000000")
            test_crossover(func, "111000001", "010110110")
            test_crossover(func, "10000101110", "00101101011")
        elif name == "mutate":
            test_mutate(func, "01100100100010100010", 0)
            test_mutate(func, "11111011110111111111", 1)
            test_mutate(func, "01001011001011100001", 0.5)
            test_mutate(func, "01001011001011100001", 0.25)
            test_mutate(func, "10100111010000101101", 0.001)
        elif name == "selectPair":
            pop1 = ["1111101100", "0000000100", "1000110010", "0010011011", "1001110000"]
            test_selectPair(func, pop1, 1)
            pop2 = ["1011011111", "0001011010", "1010110001", "1001110100", "1100001100"]
            test_selectPair(func, pop2, 2)
            pop3 = ["11111111111", "11110110110", "10101100000", "10000101000", "11101111111", "00000000000"]
            test_selectPair(func, pop3, 3)
            pop4 = ["0011111011000", "1001101110101", "0000110100110", "0010011111101", "1111011111001",
                    "0001000011110", "0111000110010", "1110011100001", "1011110010010", "0010111100111",
                    "0001110110111", "1100111011100", "0001000101000", "1001111001010", "0100101111100"]
            test_selectPair(func, pop4, 4)
        else:
            print("Sorry, I don't know how to test the function %s" % name)
    except Exception as e:
        print("\n... ERROR in %s: %s" % (name, e))
