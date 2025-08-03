import math
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy


from tron.ai.tron_model import TronModel, HeroGameState
from tron.game import GameState, Direction, GameStatus, get_possible_directions, next, get_status
from tron.ai.minimax import basic_minimax, MinimaxContext, MinimaxResult, lru_eval, cache
from tron.ai.algos import choose_direction_random

class Node:
    def __init__(
        self, 
        game_state: GameState, 
        hero_index: int, 
        is_hero: bool, 
        prev_move: Direction,
        eval: float,
        parent=None
    ):
        self.game_state = game_state
        self.hero_index = hero_index
        self.is_hero = is_hero
        self.prev_move = prev_move

        self.parent = parent
        self.children = []
        self.n_visits = 0
        self.total_reward = eval
        self.eval = eval
        self.is_expanded = False

    # Consolidate the expanding of game states to a single place
    def expand(self, mm_context: MinimaxContext):

        assert len(self.children) == 0, "Only should be expanding a node once"
        assert not self.is_expanded

        self.is_expanded = True

        if self.is_hero:

            opponent_index = 0 if self.hero_index == 1 else 1
            possible_directions = get_possible_directions(self.game_state, opponent_index)
            possible_directions = [Direction.UP] if len(possible_directions) == 0 else possible_directions

            for direction in possible_directions:
                next_dirs = [None, None]
                next_dirs[opponent_index] = direction
                next_dirs[self.hero_index] = self.prev_move

                next_game_state = next(self.game_state, next_dirs)

                eval = eval_mcts(next_game_state, is_hero=False, mm_context=mm_context)

                child_node = Node(next_game_state, self.hero_index, is_hero=False, prev_move=direction, parent=self, eval=eval)
                self.children.append(child_node)

        else:
            if get_status(self.game_state).status != GameStatus.IN_PROGRESS:
                return

            possible_directions = get_possible_directions(self.game_state, self.hero_index)

            possible_directions = [Direction.UP] if len(possible_directions) == 0 else possible_directions
            for direction in possible_directions:

                eval = eval_mcts(self.game_state, is_hero=True, mm_context=mm_context, hero_move=direction)

                child_node = Node(self.game_state, self.hero_index, is_hero=True, prev_move=direction, parent=self, eval=eval)

                self.children.append(child_node)



def eval_mcts(game_state: GameState, is_hero: bool, mm_context: MinimaxContext, hero_move=None):

    if is_hero:

        assert get_status(game_state).status == GameStatus.IN_PROGRESS
        eval = basic_minimax(game_state, depth=1, is_maximizing_player=False, context=mm_context, maximizing_player_move=hero_move).evaluation
    else:

        mm_result = basic_minimax(game_state, depth=0, is_maximizing_player=True, context=mm_context)

        # Negate hero perspective eval
        eval = -mm_result.evaluation

    return eval


def search(model: TronModel, game_state, hero_index, n_iterations, exploration_factor=2, root=None):

    if len(game_state.players) != 2:
        raise NotImplementedError()
    
    opponent_index = 0 if hero_index == 1 else 1
    mm_context = MinimaxContext(model, maximizing_player=hero_index, minimizing_player=opponent_index)

    if root is None:
        root = Node(game_state, hero_index, is_hero=False, prev_move=None, eval=0.0)

    else:
        root.parent = None


    def find_best_leaf(node) -> Node:
        while len(node.children) > 0:
            _, node = select(node)
        return node

    def select(node):
        best_value = float('-inf')
        best_action = None
        best_child = None

        #print("\nSelecting child...")
        for child in node.children:

            # IF CHILD UNVISITED ALWAYS VISIT
            # if child.n_visits == 0:
            #     ucb1_value = float('inf')  # Encourage exploration
            # else:

            # Trying without above approach, adding 1 to child.n_visits in order to avoid divide by zero
            curr_child_visits = child.n_visits + 1

            #print(f"{child.is_hero=}, {child_eval=}")

            exploitation_value = (child.total_reward / curr_child_visits)

            exploration_value = exploration_factor * np.sqrt(np.log(node.n_visits) / curr_child_visits)

            ucb1_value = exploitation_value + exploration_value
                         
            # print("Current best value: ", best_value, " current ucb1 value: ", ucb1_value)
            if ucb1_value > best_value:
                best_value = ucb1_value
                best_action = child.prev_move
                best_child = child
        return best_action, best_child

    def count_depth_of_node(node) -> int:
        count = 0
        while node.parent:
            count += 1
            node = node.parent

        return count

    def backpropagate(node):

        node.n_visits += 1
        reward = node.total_reward
        is_hero_reward = node.is_hero

        while node.parent:
            node = node.parent
            node.n_visits += 1

            if node.is_hero == is_hero_reward:
                node.total_reward += reward
            else:
                node.total_reward -= reward

    def action_probabilities(node):

        actions = []
        visits = []
        for child in node.children:
            actions.append(child.prev_move)
            visits.append(child.n_visits)

        return actions, visits


    for _ in range(root.n_visits, n_iterations):
        root.n_visits += 1
        leaf = find_best_leaf(root)

        if not leaf.is_expanded:
            leaf.expand(mm_context)
        backpropagate(leaf)

    return root
    #return action_probabilities(root)

def get_move_pair(node, hero_index, temp=1.0):

    opponent_index = 0 if hero_index == 1 else 1

    if len(node.children) == 0:
        return (choose_direction_random(node.game_state, hero_index), choose_direction_random(node.game_state, opponent_index), None)

    actions = []
    visits = []

    for child in node.children:
        actions.append(child.prev_move)
        visits.append(child.n_visits)


    assert sum(visits) > 0
    # print(f"\n\nHero:")
    # for action, visit_count, child in zip(actions, visits, node.children):

    #     print(f"{action.name:<12} {visit_count:<5} {round(child.total_reward, 3):<12}")  # left-align, width 12

    chosen_child_index = softmax_sample(visits, temp=temp)
    hero_dir = actions[chosen_child_index]

    child_node = node.children[chosen_child_index]


    actions = []
    visits = []

    for grandchild in child_node.children:
        actions.append(grandchild.prev_move)
        visits.append(grandchild.n_visits)

    if len(child_node.children) == 0 or sum(visits) == 0:
        return (hero_dir, choose_direction_random(node.game_state, opponent_index), None)

    # print(f"\n\nOpponent:")
    # for action, visit_count, child in zip(actions, visits, child_node.children):

    #     print(f"{action.name:<12} {visit_count:<5} {round(child.total_reward, 3):<12}")  # left-align, width 12


    chosen_grandchild_index = softmax_sample(visits, temp=temp)
    grandchild = child_node.children[chosen_grandchild_index]
    opponent_dir = actions[chosen_grandchild_index] 


    return (hero_dir, opponent_dir, grandchild)

def softmax_sample(visits: list[int], temp: float = 1.0) -> int:
    """
    Returns (chosen_index, probabilities) where
        probabilities = softmax(logits / T)

    temp  > 1.0  → flatter distribution (more exploration)  
    temp  < 1.0  → sharper distribution (more greedy)
    """
    if temp < 0:
        raise ValueError("Temperature must be non-negative")
    
    if temp == 0:
        
        return visits.index(max(visits))  

    # numerically stable soft-max
    # max_logit = max(logits)
    # exps = [math.exp((l - max_logit) / temp) for l in logits]
    # total = sum(exps)
    # probs = [e / total for e in exps]

    total_visits = sum(visits)
    probs = [v / total_visits for v in visits]


    exp = [p ** (1.0 / temp) for p in probs]
    sum_exp = sum(exp)
    temp_probs = [e / sum_exp for e in exp]

    # print(f"Before temp {probs=}")
    #print(f"Applied {temp=}, {temp_probs=}")
    # randomly pick an index according to these probabilities
    idx = random.choices(range(len(visits)), weights=temp_probs, k=1)[0]

    
    #print(f"Picked: {idx}")
    return idx


if __name__ == "__main__":
    print()