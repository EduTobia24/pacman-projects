# multi_agents.py
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


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
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


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        
        "*** YOUR CODE HERE ***"
        if successor_game_state.is_win():
            return 100000
        elif successor_game_state.is_lose():
            return -100000
        
        food_list = new_food.as_list()

        food_score = 0
        scared_score = 0
        ghost_score = 0
        for food in food_list:
            food_score += 1/manhattan_distance(new_pos,food)
        food_score /= len(food_list)
        for ghost in new_ghost_states:
            if ghost.scared_timer:
                scared_score += ghost.scared_timer/manhattan_distance(new_pos,ghost.get_position())
            else:
                ghost_score -= 1/manhattan_distance(new_pos, ghost.get_position())

        score = successor_game_state.get_score() + food_score + 2*ghost_score + 10*scared_score
        return score

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

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

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    #@profile
    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"      
        #returns minimax evaluation
        #@profile
        def minimax(depth, gamestate):
            num_agents = game_state.get_num_agents()
            # base case: depth limit or win or lose
            if (depth == self.depth * num_agents) or gamestate.is_win() or gamestate.is_lose():
                return self.evaluation_function(gamestate)

            # recursive case
            
            #identify the agent
            agent_idx = depth % num_agents 

            actions = gamestate.get_legal_actions(agent_idx)

            # compute and store the recursive minimax evaluation for each action
            evals = []
            for action in actions:
                evals.append(minimax(depth + 1, gamestate.generate_successor(agent_idx, action)))

            # return max eval for pacman
            if agent_idx == 0:
                return max(evals)
            # return min eval for ghost
            else: 
                return min(evals)
                    
        # starting node ( agent = 0, for minimax= 1 ( first ghost ))
        agent = 0
        actions = game_state.get_legal_actions(agent)
        max_score = -float('inf')
        for action in actions:
            score = minimax(1, game_state.generate_successor(agent, action))
            if score >= max_score:
                max_score = score
                max_action = action

        return max_action


        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"

        class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):

        num_agents = game_state.get_num_agents()

        #we initialize alpha & beta to their values
        alpha = -float('inf')
        beta = float('inf')

        best_action = None
        best_value = -float('inf') #as packman maximises, we initialise it to -infinite

        actions = game_state.get_legal_actions(0) #packmant is agent 0

        for action in actions:
            successor = game_state.generate_successor(0, action)

            value = self.alphabeta(1, successor, alpha, beta)

            if best_value < value:
                best_value = value
                best_action = action

            # update alpha at the root
            if best_value > alpha:
                alpha = best_value

        return best_action


    # ---------------------------
    #   RECURSIVE ALPHA-BETA
    # ---------------------------

    def alphabeta(self, depth, gamestate, alpha, beta):

        num_agents = gamestate.get_num_agents()

        # terminal / depth cutoff
        if depth == self.depth * num_agents or gamestate.is_win() or gamestate.is_lose():
            return self.evaluation_function(gamestate)

        # which agent plays?
        agent_idx = depth % num_agents
        actions = gamestate.get_legal_actions(agent_idx)

        # if no actions, treat like terminal
        if not actions:
            return self.evaluation_function(gamestate)


        # PACMAN (MAX)
        
        if agent_idx == 0:
            value = -float('inf')

            for action in actions:
                successor = gamestate.generate_successor(agent_idx, action)

                v = self.alphabeta(depth + 1, successor,
                                   max(alpha, value),  # update alpha going down
                                   beta)

                if v > value:
                    value = v

                # PRUNE (strict, no equality pruning)
                if value > beta:
                    return value

            return value

        # GHOSTS (MIN)

        else:
            value = float('inf')

            for action in actions:
                successor = gamestate.generate_successor(agent_idx, action)

                v = self.alphabeta(depth + 1, successor, alpha, min(beta, value))  # update beta going down

                if v < value:
                    value = v

                # PRUNE (strict)
                if value < alpha:
                    return value

            return value


        


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
