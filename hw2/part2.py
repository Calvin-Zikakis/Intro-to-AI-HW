import robby
#import numpy as np
from utils import *
import numpy as np
import random
POSSIBLE_ACTIONS = ["MoveNorth", "MoveSouth", "MoveEast", "MoveWest", "StayPut", "PickUpCan", "MoveRandom"]
rw = robby.World(10, 10)
rw.graphicsOff(message="")


def sortByFitness(genomes):
    tuples = [(fitness(g), g) for g in genomes]
    tuples.sort()
    sortedFitnessValues = [f for (f, g) in tuples]
    sortedGenomes = [g for (f, g) in tuples]
    return sortedGenomes, sortedFitnessValues


def randomGenome(length):
    """
    :param length:
    :return: string, random integers between 0 and 6 inclusive
    """

    """Your Code Here"""
    bitstring = ""
    for i in range(length):
        random_choice = np.random.randint(0,7)
        bitstring = bitstring + str(random_choice)
    return bitstring

def makePopulation(size, length):
    """
    :param size - of population:
    :param length - of genome
    :return: list of length size containing genomes of length length
    """


    """Your Code Here"""
    population = []
    i = 0
    while i < size:
        population.append(randomGenome(length))
        #loop over size to get amount of genomes required. Build genomes and append to list
        i+=1
    return population

def fitness(genome, steps=200, init=0.50):
    """

    :param genome: to test
    :param steps: number of steps in the cleaning session
    :param init: amount of cans
    :return:
    """
    reward = 0
    
    for i in range(25):
    #loop over 25 times

        if type(genome) is not str or len(genome) != 243:
            raise Exception("strategy is not a string of length 243")
        for char in genome:
            if char not in "0123456":
                raise Exception("strategy contains a bad character: '%s'" % char)
        if type(steps) is not int or steps < 1:
            raise Exception("steps must be an integer > 0")
        if type(init) is str:
            # init is a config file
            rw.load(init)
        elif type(init) in [int, float] and 0 <= init <= 1:
            # init is a can density
            rw.goto(0, 0)
            rw.distributeCans(init)
        else:
            raise Exception("invalid initial configuration")

        for x in range(steps):

            p = rw.getPerceptCode()
            action = POSSIBLE_ACTIONS[int(genome[p])]
            reward = rw.performAction(action) + reward
    return reward / 25




def evaluateFitness(population):
    """
    :param population:
    :return: a pair of values: the average fitness of the population as a whole and the fitness of the best individual
    in the population.
    """
    current_best = -999999
    total_fitness = 0
    amount = 0

    for count, item in enumerate(population):
        gene = fitness(item)
        total_fitness = total_fitness + gene
        if (gene > current_best):
            current_best = gene
        amount = count

    average = float(total_fitness)/(amount+1)
    return average, current_best


def crossover(genome1, genome2):
    """
    :param genome1:
    :param genome2:
    :return: two new genomes produced by crossing over the given genomes at a random crossover point.
    """
    random_point = np.random.randint(1, len(genome1))

    child_1 = genome1[:random_point] + genome2[random_point:]
    child_2 = genome2[:random_point] + genome1[random_point:]
    return child_1, child_2


def mutate(genome, mutationRate):
    """
    :param genome:
    :param mutationRate:
    :return: a new mutated version of the given genome.
    """
    genome_list = list(genome)
    mutated_list = []
    if mutationRate == 0:
        #no mutation
        return genome

    elif mutationRate != 0:
        #some mutation

        amount_to_mutate = int(len(genome) * mutationRate)
        #how many bits are we mutating

        mutations = np.random.choice(len(genome), amount_to_mutate, replace=False)
        #randomly choose what to mutate

        for i in mutations:
            random_value = np.random.randint(0,7)
            pre_mutation = genome_list[i]
            if pre_mutation == random_value:
                    if random_value == 6:
                        random_value = random_value - 1
                    else:
                        random_value = random_value + 1
            genome_list[i] = random_value

    return (''.join([str(elem) for elem in genome_list]))
            




def selectPair(population):
    """

    :param population:
    :return: two genomes from the given population using fitness-proportionate selection.
    This function should use RankSelection,
    """
    '''
    sortedGenomes, sortedFitnessValues = sortByFitness(population)
    return sortedGenomes[0], sortedGenomes[1]

'''
    weights = []
    for i in range(len(population)):
        weights.append(i)
    
    choice = weightedChoice(population, weights)
    choice2 = weightedChoice(population, weights)

    return choice, choice2

    


def runGA(populationSize, crossoverRate, mutationRate, logFile=""):
    """

    :param populationSize: :param crossoverRate: :param mutationRate: :param logFile: :return: xt file in which to
    store the data generated by the GA, for plotting purposes. When the GA terminates, this function should return
    the generation at which the string of all ones was found.is the main GA program, which takes the population size,
    crossover rate (pc), and mutation rate (pm) as parameters. The optional logFile parameter is a string specifying
    the name of a te
    """
    s=0
    save = True
    if logFile == "":
        save = False
    #if we need to save the file, save it.

    if save:
        savefile = open(logFile, 'a')
    if save:
        savefile2 = open("Logfilegraphing2.txt", 'a')

    genomeLength = 243

    print("Population size: ", populationSize)
    print("Genome length: ", genomeLength)

    population = makePopulation(populationSize, genomeLength)
    generation = 0

    x = 0

    while x <= 300:
        newGeneration = []

        avg, highest = evaluateFitness(population)

        population, fitness_values = sortByFitness(population)

        print("Generation   ", generation, ":", "average fitness ", avg, ", ", " best fitness ", highest)
        if save and s % 10 == 0:
            savefile.write(str(generation)+" "+str(avg)+" "+str(highest)+" "+str(population[99])+"\n")
        if save:
            savefile2.write(str(avg)+", ")
    

        for i in range(int(populationSize/2)):
            childA, childB = selectPair(population)

            if(random.random() < crossoverRate):
                childA, childB = crossover(childA, childB)
            


            childA = mutate(childA, mutationRate)
            childB = mutate(childB, mutationRate)


            newGeneration.append(childA)
            newGeneration.append(childB)


        population = newGeneration


        generation = generation + 1

        x = x + 1
        s = s + 1

    if save:
        savefile.close()
        savefile2.close()
    return generation



def test_FitnessFunction():
    f = fitness(rw.strategyM)
    print("Fitness for StrategyM : {0}".format(f))



#test_FitnessFunction()

runGA(100, 1, 0.005,"bestStrategy4.txt")

