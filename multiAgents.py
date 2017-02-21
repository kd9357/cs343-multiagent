# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        v = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == v]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodList = newFood.asList()
        if len(foodList) == 0:
          return 100000
        if currentGameState.getPacmanPosition() == newPos:
          return -100000
        foodScore = 1.0 / min([manhattanDistance(newPos, food) for food in foodList])
        for i in range(0, len(newGhostStates)):
          if newScaredTimes[i] == 0 and manhattanDistance(newPos, newGhostStates[i].getPosition()) <= 1:
            return -100000
        return successorGameState.getScore() + foodScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth):
          if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
          actions = gameState.getLegalActions(0)
          v = -100000
          for action in actions:
            newState = gameState.generateSuccessor(0, action)
            v = max(v, minValue(newState, depth, 1))
          return v

        def minValue(gameState, depth, agent):
          if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
          actions = gameState.getLegalActions(agent)
          v = 100000
          for action in actions:
            newState = gameState.generateSuccessor(agent, action)
            if agent == gameState.getNumAgents() - 1:
              v = min(v, maxValue(newState, depth + 1))
            else:
              v = min(v, minValue(newState, depth, agent + 1))
          return v

        actions = gameState.getLegalActions()
        direction = Directions.STOP
        v = -100000
        for action in actions:
          newState = gameState.generateSuccessor(0, action)
          newScore = minValue(newState, 0, 1)
          if newScore > v:
            v = newScore
            direction = action
        return direction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth, alpha, beta):
          if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
          v = -100000
          actions = gameState.getLegalActions(0)
          for action in actions:
            newState = gameState.generateSuccessor(0, action)
            v = max(v, minValue(newState, depth, 1, alpha, beta))        
            if v > beta:
              return v
            alpha = max(alpha, v)
          return v

        def minValue(gameState, depth, agent, alpha, beta):
          if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
          v = 100000
          actions = gameState.getLegalActions(agent)
          for action in actions:
            newState = gameState.generateSuccessor(agent, action)
            if agent == gameState.getNumAgents() - 1:
              v = min(v, maxValue(newState, depth + 1, alpha, beta))
            else:
              v = min(v, minValue(newState, depth, agent + 1, alpha, beta))            
            if alpha > v:
              return v
            beta = min(beta, v)
          return v

        actions = gameState.getLegalActions()
        direction = Directions.STOP
        v = -100000
        alpha = -100000
        beta = 100000
        for action in actions:
          newState = gameState.generateSuccessor(0, action)
          newScore = minValue(newState, 0, 1, alpha, beta)
          if newScore > v:
            v = newScore
            direction = action
          if v > beta:
            return direction
          alpha = max(alpha, v);
        return direction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth):
          if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
          v = -100000
          actions = gameState.getLegalActions(0)
          for action in actions:
            newState = gameState.generateSuccessor(0, action)
            v = max(v, expectedValue(newState, depth, 1))
          return v
        def expectedValue(gameState, depth, agent):
          if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
          v = 0
          actions = gameState.getLegalActions(agent)
          probability = len(actions)
          for action in actions:
            newState = gameState.generateSuccessor(agent, action)
            if agent == gameState.getNumAgents() - 1:
              v += maxValue(newState, depth + 1)
            else:
              v += expectedValue(newState, depth, agent + 1)
          return v / probability

        actions = gameState.getLegalActions(0)
        direction = Directions.STOP
        v = -100000.0
        for action in actions:
          newState = gameState.generateSuccessor(0, action)
          newScore = max(v, expectedValue(newState, 0, 1))
          if newScore > v:
            v = newScore
            direction = action
        return direction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      If the game is a win or loss, immediately return
      Otherwise calculate the closest food and add the inverse distance to the score
      Punish idling by subtracting remaining food pellets from the score heavily
      If the ghosts are unafraid, reward the agent by maintaining a distance from the closest
        ghost of at least 3
      If the ghosts are afraid, reward the agent for being near the ghosts and add remaining
        time to the score
      Finally, weight the reward of food and the reward for ghosts separately, and combine
        with the currentGame score
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
      return 100000
    if currentGameState.isLose():
      return -100000

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    foodDist = [manhattanDistance(pos, food) for food in foodList]

    ghostStates = currentGameState.getGhostStates()
    scaredTime = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPos = currentGameState.getGhostPositions()
    ghostDist = [manhattanDistance(pos, ghost) for ghost in ghostPos]

    foodScore = 0.0
    ghostScore = 0.0

    foodScore = 1.0 / min(foodDist)
    foodScore -= 4 * len(foodList)
    if sum(scaredTime) <= 1: 
      #Escape
      ghostScore += max(min(ghostDist), 3)
    else:
      #Hunt
      for ghost in ghostDist:
        ghostScore += 1.0 / ghost
    for time in scaredTime:
      ghostScore += time
    return currentGameState.getScore() + foodScore * 3 + ghostScore * 0.4
# Abbreviation
better = betterEvaluationFunction

