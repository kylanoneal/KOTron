from AI.pytorch_game_utils import *
from game.KyTron import *

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

    # Consolidate the expanding of game states to a single place
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


def search(model, game, player_num, head_val, n_iterations, exploration_factor=2):
    model_type = type(model)
    root = Node(game, player_num)
    def find_best_leaf(node) -> Node:
        while node.is_expanded and not node.is_terminal:
            _, node = select(node)
        return node

    def select(node):
        best_value = float('-inf')
        best_action = None
        best_child = None
        # print("about to expand children")
        # print("node children length:", len(node.children.items()))
        for action, child in node.children.items():
            ## POSSIBLE JANK

            # IF CHILD UNVISITED ALWAYS VISIT
            # if child.n_visits == 0:
            #     ucb1_value = float('inf')  # Encourage exploration
            # else:

            # Trying without above approach, adding 1 to child.n_visits in order to avoid divide by zero
            curr_child_visits = child.n_visits + 1

            grid, heads = get_relevant_info_from_game_state(child.state)
            child_eval = model(get_model_input_from_raw_info(grid, heads, child.player_num,
                                                             head_val, model_type=model_type))

            ucb1_value = (child.total_reward / curr_child_visits) + child_eval + \
                         np.sqrt(exploration_factor * np.log(node.n_visits) / curr_child_visits)
            # print("Current best value: ", best_value, " current ucb1 value: ", ucb1_value)
            if ucb1_value > best_value:
                best_value = ucb1_value
                best_action = action
                best_child = child
        return best_action, best_child

    def count_depth_of_node(node) -> int:
        count = 0
        while node.parent:
            count += 1
            node = node.parent

        return count

    def backpropagate(node, value):
        if node.parent:
            node.total_reward += value - node.value
            node.n_visits += 1
            backpropagate(node.parent, value)

    def action_probabilities(node):

        actions = []
        visits = []
        for action, n_visits in node.children.items():
            actions.append(action)
            visits.append(n_visits)
            

        return actions, visits


    for _ in range(n_iterations):
        root.n_visits += 1
        leaf = find_best_leaf(root)
        grid, heads = get_relevant_info_from_game_state(leaf.state)
        eval = model(get_model_input_from_raw_info(grid, heads, leaf.player_num, model_type=model_type))

        leaf.expand(eval)
        backpropagate(leaf, eval)

    return action_probabilities(root)

