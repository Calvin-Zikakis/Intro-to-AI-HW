# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def pathlocation(options):
    location = []
    for i in options:
        location.append(i[1])
        return location


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    frontier = util.Stack()
    frontier.push((problem.getStartState(),[]))

    visited = set()

    while not frontier.isEmpty():
        s, a = frontier.pop()

        if problem.isGoalState(s):
            return a
        
        visited.add(s)

        for i in problem.getSuccessors(s):
            if i[0] not in frontier.list and i[0] not in visited:
                frontier.push((i[0], a + [i[1]]))



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    frontier.push((problem.getStartState(),[]))

    visited = set()

    while not frontier.isEmpty():
        s, a = frontier.pop()

        for n in problem.getSuccessors(s):
            n_d= n[1]
            n_s = n[0]
            if n_s not in visited:
                if problem.isGoalState(n_s):
                    return a + [n_d]
                frontier.push((n_s, a + [n_d]))
                visited.add(n_s)



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    x = (problem.getStartState(),0)
    c = x[1]
    n = x[0]
    visited = set()

    frontier = util.PriorityQueue()
    frontier.push((n,[],0),0)

    while not frontier.isEmpty():
        n, p, c  = frontier.pop()
        visited.add(n)

        if problem.isGoalState(n):
            return p

        for y in problem.getSuccessors(n):
            c_n= y[0]
            c_p = y[1]
            c_c = y[2]
            if c_n not in visited:
                if c_n not in frontier.heap:
                    frontier.push((c_n, p + [c_p], c_c + c), c + c_c)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    x = (problem.getStartState(),0)
    c = x[1]
    n = x[0]
    visited = set()

    frontier = util.PriorityQueue()
    frontier.push((n,[],0),0)

    while not frontier.isEmpty():
        n, p, c  = frontier.pop()
        visited.add(n)

        if problem.isGoalState(n):
            return p

        for y in problem.getSuccessors(n):
            c_n= y[0]
            c_p = y[1]
            c_c = y[2]
            if c_n not in visited:
                if c_n not in frontier.heap:
                    frontier.push((c_n, p+[c_p], c_c + c), c + c_c + heuristic(c_n, problem))



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
