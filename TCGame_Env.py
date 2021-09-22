from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.NaN for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        
        winning_idx_lists = [
                                [0, 1, 2],
                                [3, 4, 5],
                                [6, 7, 8],
                                [0, 3, 6],
                                [1, 4, 7],
                                [2, 5, 8],
                                [0, 4, 8],
                                [2, 4, 6]
                            ]
        is_winning = False
        for winning_idx_list in winning_idx_lists:
            result_list = [curr_state[index] for index in winning_idx_list]
            if (sum(result_list) == 15):
                is_winning = True 
                break

        return is_winning
 

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) == 0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""
        # Action space - allowed (position, value) combinations for the agent and environment given the current state

        agent_actions = list(product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0]))
        env_actions = list(product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1]))
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        next_state = curr_state.copy()
        next_state[curr_action[0]] = curr_action[1]
        return next_state

    def get_reward(self, is_terminal_state, result, reward, is_agent_move):
        """ If your agent wins the game, it gets 10 points,
            if the environment wins, the agent loses 10 points.
            And if the game ends in a draw, it gets 0.
            Also, for each move, it gets a -1 point.
        """
        if is_terminal_state:
            reward += 0 if result == "Tie" else 10 if is_agent_move else -10
        elif (not is_terminal_state and is_agent_move):
            reward += -1

        return reward

    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        # Perform agent's move
        next_state = self.state_transition(curr_state, curr_action)
        
        # Check if terminal state reached
        has_reached_terminal, result = self.is_terminal(next_state)
        
        # Get the reward based on the agent move
        reward = self.get_reward(has_reached_terminal, result, reward = 0, is_agent_move = True)
        
        if not has_reached_terminal:
            # Check the possible moves for the environment
            _, env_space = self.action_space(next_state)
            
            # Pick a random action from environment space
            env_action = random.choice(env_space)
            
            # Perform environment's move
            next_state = self.state_transition(next_state, env_action)

            # Check if terminal state reached
            has_reached_terminal, result = self.is_terminal(next_state)

            # Update the reward based on the environment's move
            reward = self.get_reward(has_reached_terminal, result, reward, is_agent_move = False)
        
        # Return current state, reward, has reached terminal
        return next_state, reward, has_reached_terminal

    def reset(self):
        return self.state
