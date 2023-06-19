import numpy as np
from copy import deepcopy
from game.BMTron import *
from AI.pytorch_models import *
#from pytorch_models import get_model_input_from_game_state

EQUAL_ACTION_PROBS = np.array([0.25, 0.25, 0.25, 0.25])
class Node:
    def __init__(self, state: BMTron, player_num, parent=None):
        self.state = state
        self.player_num = player_num
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.total_reward = 0.0
        self.is_expanded = False
        self.is_terminal = False
        self.value = 0

    def expand(self, value):
        self.is_expanded = True
        self.value = value

        available_actions = self.state.get_possible_directions(self.player_num)

        # print("I: ", i, "action: ", action)
        for action in available_actions:
            next_game_state = deepcopy(self.state)
            next_game_state.update_direction(self.player_num, action)
            next_game_state.move_racers()

            self.children[action] = Node(next_game_state, self.player_num, parent=self)

        self.is_terminal = not self.children

class MCTS:
    def __init__(self, model, n_simulations=10, exploration_factor=2):
        self.model = model
        self.model_type = type(model)
        self.n_simulations = n_simulations
        self.exploration_factor = exploration_factor

    def search(self, root: Node):

        for _ in range(self.n_simulations):
            root.n_visits += 1
            leaf = self.find_best_leaf(root)
            #print("Depth of leaf:", self.count_depth_of_node(leaf))

            grid, heads = get_relevant_info_from_game_state(leaf.state)
            eval = self.model(get_model_input_from_raw_info(grid, heads, leaf.player_num, model_type=self.model_type))

            leaf.expand(eval)
            self.backpropagate(leaf, eval)
        return self.action_probabilities(root)

    def find_best_leaf(self, node) -> Node:
        while node.is_expanded and not node.is_terminal:
            _, node = self.select(node)
        return node

    def select(self, node):
        best_value = float('-inf')
        best_action = None
        best_child = None
        #print("about to expand children")
        #print("node children length:", len(node.children.items()))
        for action, child in node.children.items():
            ## POSSIBLE JANK

            #IF CHILD UNVISITED ALWAYS VISIT
            # if child.n_visits == 0:
            #     ucb1_value = float('inf')  # Encourage exploration
            # else:

            #Trying without above approach, adding 1 to child.n_visits in order to avoid divide by zero
            curr_child_visits = child.n_visits + 1

            grid, heads = get_relevant_info_from_game_state(child.state)
            child_eval = self.model(get_model_input_from_raw_info(grid, heads, child.player_num, model_type=self.model_type))


            ucb1_value = (child.total_reward / curr_child_visits) + child_eval+ \
                         np.sqrt(self.exploration_factor * np.log(node.n_visits) / curr_child_visits)
            #print("Current best value: ", best_value, " current ucb1 value: ", ucb1_value)
            if ucb1_value > best_value:
                best_value = ucb1_value
                best_action = action
                best_child = child
        return best_action, best_child

    def count_depth_of_node(self, node) -> int:
        count = 0
        while node.parent:
            count += 1
            node = node.parent

        return count
    def backpropagate(self, node, value):
        if node.parent:
            node.total_reward += value - node.value
            node.n_visits += 1
            self.backpropagate(node.parent, value)

    def action_probabilities(self, node, temperature=0.2):
        visits = np.array([child.n_visits for child in node.children.values()])
        #print("VISITS:", visits)
        actions = list(node.children.keys())
        #print("Actions: ", actions, "\n")

        all_action_probs = np.zeros(4) if len(actions) > 0 else EQUAL_ACTION_PROBS


        if temperature == 0:  # zero temperature corresponds to taking the max
            raise NotImplementedError
            # if len(actions) > 0:
            #
            #     all_action_probs[actions[np.argmax(visits)].value] = 1
            # return actions, all_action_probs
        else:  # otherwise we apply a softmax operation
            visits = visits**(1/temperature)
            probs = visits / np.sum(visits)

            #print("MCTS probs:", probs)

            # for i in range(len(actions)):
            #     all_action_probs[actions[i].value] = probs[i]

            #print("action probs:", all_action_probs, "\n")
            return actions, probs

