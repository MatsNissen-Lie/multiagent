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


from math import sqrt
from util import manhattanDistance
from game import Actions, Directions
import random, util

from game import Agent
from pacman import GameState, GhostRules


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newGhostStates: list[GhostRules] = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score_sum = 0  # successorGameState.getScore()

        if successorGameState.isWin():
            return 99999999

        if successorGameState.isLose():
            return -99999999

        if newPos in currentGameState.getFood().asList():
            score_sum += 100

        ghost_positions = [ghost.getPosition() for ghost in newGhostStates]

        nearest_food = foodHeuristic((newPos, newFood))
        nearest_ghost = distance_heuristic(newPos, ghost_positions)

        # heuristic for the score can be very big for the food.
        score_sum += 10 / nearest_food**0.5
        score_sum -= 10 / (nearest_ghost) ** 2

        return score_sum


def foodHeuristic(state: tuple[tuple, list[list]]):
    position, foodGrid = state
    foodList = foodGrid.asList()
    return distance_heuristic(position, foodList)


def distance_heuristic(position, pos_list: list[tuple]) -> int:
    if not pos_list:
        return 0

    # Compute the distance to the nearest food pellet.
    nearest_food_dist = min(manhattanDistance(position, food) for food in pos_list)

    # Compute MST cost using Prim's algorithm with an array for minimum connection cost.
    n = len(pos_list)
    in_mst = [False] * n
    # cost_to_connect[i] will hold the minimum Manhattan distance
    # needed to connect pos_list[i] to the current MST.
    cost_to_connect = [float("inf")] * n

    # Start with the first food pellet.
    cost_to_connect[0] = 0
    mst_cost = 0

    for _ in range(n):
        # Find the unvisited node with the smallest cost_to_connect.
        best = None
        best_cost = float("inf")
        for i in range(n):
            if not in_mst[i] and cost_to_connect[i] < best_cost:
                best_cost = cost_to_connect[i]
                best = i

        # Add the selected node to the MST.
        in_mst[best] = True
        mst_cost += best_cost

        # Update the cost to connect for the remaining nodes.
        for j in range(n):
            if not in_mst[j]:
                d = manhattanDistance(pos_list[best], pos_list[j])
                if d < cost_to_connect[j]:
                    cost_to_connect[j] = d

    return nearest_food_dist + mst_cost


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            actions = gameState.getLegalActions(0)
            # The trvial situations(state)
            if (
                len(actions) == 0
                or gameState.isWin()
                or gameState.isLose()
                or depth == self.depth
            ):
                return (self.evaluationFunction(gameState), None)

            # We are trying to implement the 2 sides of the minimax algorithm the max and the min
            w = -(float("inf"))
            move = None
            # In that way that the 2 functions are calling each other is like building the tree(diagrams from tha class)
            for action in actions:
                # We have the available moves and we are seeking for the "best" one
                sucsValue, _ = min_value(
                    gameState.generateSuccessor(0, action), 1, depth
                )
                # It is working exactly as the theory of minimax algorithm commands
                if sucsValue > w:  # Here we have as start -infinite
                    w, move = sucsValue, action
            return (w, move)

        def min_value(gameState, agentID, depth):
            actions = gameState.getLegalActions(agentID)
            is_last_agent = agentID == gameState.getNumAgents() - 1
            if len(actions) == 0:
                return (self.evaluationFunction(gameState), None)
            v = float("inf")
            move = None

            for action in actions:
                if not is_last_agent:
                    successor_value, _ = min_value(
                        gameState.generateSuccessor(agentID, action), agentID + 1, depth
                    )
                else:
                    successor_value, _ = max_value(
                        gameState.generateSuccessor(agentID, action), depth + 1
                    )
                if successor_value < v:
                    v, move = successor_value, action
            return v, move

        best_action = max_value(gameState, 0)[1]
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth, alpha=-float("inf"), beta=float("inf")):
            actions = gameState.getLegalActions(0)
            # The trvial situations(state)
            if (
                len(actions) == 0
                or gameState.isWin()
                or gameState.isLose()
                or depth == self.depth
            ):
                return (self.evaluationFunction(gameState), None)

            # We are trying to implement the 2 sides of the minimax algorithm the max and the min
            w = -(float("inf"))
            move = None
            # In that way that the 2 functions are calling each other is like building the tree(diagrams from tha class)
            for action in actions:
                # We have the available moves and we are seeking for the "best" one
                sucsValue, _ = min_value(
                    gameState.generateSuccessor(0, action), 1, depth, alpha, beta
                )
                # It is working exactly as the theory of minimax algorithm commands
                if sucsValue > w:  # Here we have as start -infinite
                    w, move = sucsValue, action
                if w > beta:
                    return (w, move)
                alpha = max(alpha, w)
            return (w, move)

        def min_value(
            gameState, agentID, depth, alpha=-float("inf"), beta=float("inf")
        ):
            actions = gameState.getLegalActions(agentID)
            is_last_agent = agentID == gameState.getNumAgents() - 1
            if len(actions) == 0:
                return (self.evaluationFunction(gameState), None)
            v = float("inf")
            move = None

            for action in actions:
                if not is_last_agent:
                    successor_value, _ = min_value(
                        gameState.generateSuccessor(agentID, action),
                        agentID + 1,
                        depth,
                        alpha,
                        beta,
                    )
                else:
                    successor_value, _ = max_value(
                        gameState.generateSuccessor(agentID, action),
                        depth + 1,
                        alpha,
                        beta,
                    )
                if successor_value < v:
                    v, move = successor_value, action
                if v < alpha:
                    return (v, move)
                beta = min(beta, v)
            return v, move

        best_action = max_value(gameState, 0)[1]
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            """Return (value, action) for Pacman (agent 0)."""
            actions = gameState.getLegalActions(0)
            # Check for terminal states or if maximum search depth is reached.
            if (
                len(actions) == 0
                or gameState.isWin()
                or gameState.isLose()
                or depth == self.depth
            ):
                return (self.evaluationFunction(gameState), None)

            best_value = -float("inf")
            best_action = None
            for action in actions:
                # Generate successor for Pacman’s action.
                successor = gameState.generateSuccessor(0, action)
                # Next, go to the first ghost's turn.
                value = exp_value(successor, 1, depth)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action

        def exp_value(gameState, agentID, depth):
            """Return the expected value for a ghost agent."""
            actions = gameState.getLegalActions(agentID)
            if len(actions) == 0:
                return self.evaluationFunction(gameState)

            is_last_agent = agentID == gameState.getNumAgents() - 1
            total_value = 0
            probability = 1 / len(actions)  # Uniform probability over legal actions.
            for action in actions:
                successor = gameState.generateSuccessor(agentID, action)
                # If this ghost is not the last agent, then move to the next ghost.
                if not is_last_agent:
                    value = exp_value(successor, agentID + 1, depth)
                else:
                    # Last ghost; the next turn is Pacman's (maximizer) turn.
                    value = max_value(successor, depth + 1)[0]
                total_value += value * probability
            return total_value

        # Start expectimax search with Pacman (agent 0) at depth 0.
        best_action = max_value(gameState, 0)[1]
        return best_action


class SearchProblem:
    def __init__(self, gameState: GameState):
        self.startingState = gameState.getPacmanPosition()
        self.goalStates = gameState.getFood().asList()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._expanded = 0

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        return self.startingState

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        x, y = state
        return (x, y) in self.goalStates

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        successors = []
        for action in [
            Directions.NORTH,
            Directions.SOUTH,
            Directions.EAST,
            Directions.WEST,
        ]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))
        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        x, y = self.startingState
        cost = 0
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            cost += self.costFn((x, y))
        return cost


def breadthFirstSearch(problem: SearchProblem) -> list[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()
    frontier = util.Queue()
    frontier.push((root, []))
    explored = set()

    while not frontier.isEmpty():
        current, actions = frontier.pop()
        if problem.isGoalState(current):
            return actions
        if current not in explored:
            explored.add(current)
            for successor, action, _ in problem.getSuccessors(current):
                frontier.push((successor, actions + [action]))
    return []


def betterEvaluationFunction(currentGameState: GameState):
    """
    This evaluation function rewards Pacman for progress (eating food and capsules)
    and avoiding ghosts, and it also penalizes Pacman for standing still.
    """
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    # Start with the current game score
    score = currentGameState.getScore()

    # 1) Distance to the closest food
    if len(foodList) > 0:
        minFoodDist = distance_heuristic(pacmanPos, foodList)
    else:
        minFoodDist = 0

    # 2) Distance to the closest capsule
    if len(capsules) > 0:
        minCapsuleDist = min(manhattanDistance(pacmanPos, cap) for cap in capsules)
        # minCapsuleDist = distance_heuristic(pacmanPos, capsules)
    else:
        minCapsuleDist = 0

    # 3) Ghost distances: penalize active ghosts if they're close;
    #    encourage chasing if they're scared.
    ghostPenalty = 0
    for ghost in ghostStates:
        dist = manhattanDistance(pacmanPos, ghost.getPosition())
        if dist == 0:
            # If ghost is on the same spot and not scared -> very bad!
            if ghost.scaredTimer == 0:
                ghostPenalty += 1000
        else:
            if ghost.scaredTimer > 1:
                # Encourage chasing scared ghosts (closer is better).
                ghostPenalty -= 300.0 / dist
            else:
                # If ghost is active and too close, penalize more.
                if dist < 2:
                    ghostPenalty += 500
                else:
                    ghostPenalty += 2.0 / dist

    # 4) Penalize Pacman if he's standing still.
    pacmanDirection = currentGameState.getPacmanState().configuration.direction
    stopPenalty = 0
    if pacmanDirection == Directions.STOP:
        stopPenalty = 500  # Increase this if you want to penalize more

    finalScore = (
        score
        - 1.5 * minFoodDist
        - 1.0 * minCapsuleDist
        - 10.0 * len(foodList)
        - 20.0 * len(capsules)
        - ghostPenalty
        - stopPenalty * 4
    )

    return finalScore


# Abbreviation
better = betterEvaluationFunction
