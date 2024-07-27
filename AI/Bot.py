import copy
import random
from AI.AStar import does_path_exist
from game.ko_tron import KOTron, Directions
from AI.pytorch_game_utils import get_model, get_next_action, get_random_next_action


class Evaluation:

    def __init__(self, self_evaluation, opponent_evaluation=None):
        self.self_evaluation = self_evaluation
        self.opponent_evaluation = opponent_evaluation

        if opponent_evaluation is not None:

            self.score = (1 - opponent_evaluation.score + self_evaluation.score) / 2

        else:

            self.score = self_evaluation.score


class PlayerEvaluation:

    def __init__(self, density_max_tuple, distance_subscore, path_subscore, target_square, grid=None,
                 quadrant_all_weight=False):
        self.target_square = target_square
        self.grid = grid
        self.density, self.max = density_max_tuple
        self.density_subscore = 1 - self.density / self.max
        self.distance_subscore = distance_subscore
        self.path_subscore = path_subscore

        if quadrant_all_weight:
            path_weight = 1
            density_weight = 0
            distance_weight = 0

        else:
            path_weight = 0
            density_weight = .5
            distance_weight = .5

        self.score = self.density_subscore * density_weight + distance_subscore * distance_weight + path_subscore * path_weight

    @staticmethod
    def mininmum_score():
        return PlayerEvaluation((1, 1), 0, 0, (0, 0), False)


class BotConfig:

    def __init__(self, look_ahead, depth=1, consider_opponent=True, quadrant_all_weight=False):
        self.quadrant_all_weight = quadrant_all_weight
        self.consider_opponent = consider_opponent
        self.depth = depth
        self.look_ahead = look_ahead


class TronBot:
    """The only internal state is the game state"""
    def __init__(self, game: KOTron, player_num):
        self.game = game
        self.player_num = player_num

    def bot_move(self) -> Directions:
        """Pick the move to use based on the current game state"""
        raise NotImplementedError()

class ReinforcementBot(TronBot):

    def __init__(self, game, player_num, model_path):
        super().__init__(game, player_num)
        self.model = get_model(model_path)

    def bot_move(self):
        action = get_next_action(self.model, self.game, self.player_num, temperature=0.0)
        self.game.update_direction(self.player_num, action)

class RandomBot(TronBot):
    def __init__(self, game, player_num):
        super().__init__(game, player_num)

    def bot_move(self):
        action = get_random_next_action(self.game, self.player_num)
        self.game.update_direction(self.player_num, action)

class MiniMaxBot(TronBot):

    def __init__(self, game):
        super().__init__(game)

    def bot_move(self):

        # Change to not hard code player num
        direction = self.choose_bot_move(0, BotConfig(look_ahead=1, depth=1, quadrant_all_weight=True))

        if direction is not None:
            self.game.update_direction(1, direction)

    def old_choose_bot_move(self, player_index):
        heads = self.game.get_heads()
        possible_directions = self.game.get_possible_directions(player_index)

        if len(possible_directions) > 0:
            files = self.get_distances(self.game, player_index, possible_directions)
            open_file = possible_directions[files.index(max(files))]
            return self.choose_direction(possible_directions, open_file, player_index)

        return None

    def minimax_evaluate(self, game_state, player_index, bot_config, depth):

        if depth == 0:
            return self.evaluate(game_state, player_index, bot_config), None
        else:

            moves = []

            possible_directions = game_state.get_possible_directions(player_index)

            if len(possible_directions) == 0:
                return self.evaluate(game_state, player_index, bot_config), None
            for i in range(len(possible_directions)):

                candidate_state = copy.deepcopy(game_state)

                candidate_state.update_direction(player_index + 1, possible_directions[i])

                for i in range(bot_config.look_ahead):
                    candidate_state.move_racers()

                move = self.minimax_evaluate(candidate_state, player_index, bot_config, depth - 1)
                assert isinstance(move[0], Evaluation)
                moves.append(move)

            scores = [move[0].score for move in moves]
            max_score = max(scores)
            if depth == 2:
                self.print_scores(moves, possible_directions)

                print("\n")
            index = scores.index(max_score)
            return moves[index][0], possible_directions[index]

    def choose_bot_move(self, player_index, bot_config):

        return self.minimax_evaluate(self.game, player_index, bot_config, bot_config.depth)[1]

    def evaluate_position(self, game_state, player_index, bot_config):
        possible_directions = game_state.get_possible_directions(player_index)

        if len(possible_directions) == 0:
            return PlayerEvaluation.mininmum_score()

        sum_distances = sum(self.get_distances(game_state, player_index, possible_directions))
        distance_subscore = sum_distances / 100

        density, max, grid = self.calc_density(game_state, player_index)

        result = self.calculate_path_existence(game_state, player_index)
        target_square = result[1]
        path_subscore = 1 if result[0][0] else 0

        return PlayerEvaluation((density, max), distance_subscore, path_subscore, target_square, grid,
                                bot_config.quadrant_all_weight)

    def calculate_path_existence(self, game_state, player_index):
        increment = game_state.dimension / 4
        big_increment = increment * 3
        quadrant = self.get_quadrant(game_state, player_index)
        goals = [(increment, big_increment), (big_increment, big_increment), \
                 (big_increment, increment), (increment, increment)]

        return does_path_exist(tuple(game_state.players[player_index].head), goals[quadrant - 1],
                               game_state.collision_table), goals[quadrant - 1]

    def get_quadrant(self, game_state, player_index):
        half = game_state.dimension / 2
        x, y = game_state.players[player_index].head

        if x > half:

            if y > half:
                return 4
            else:
                return 1
        else:

            if y > half:
                return 3
            else:
                return 2

    def evaluate(self, game_state, player_index, bot_config):

        other_player_index = 1 if player_index == 0 else 0
        if bot_config.consider_opponent:

            return Evaluation(self.evaluate_position(game_state, player_index, bot_config), \
                              self.evaluate_position(game_state, other_player_index, bot_config))

        else:

            return Evaluation(self.evaluate_position(game_state, player_index, bot_config))

    def calc_density(self, game_state, player_index):

        bot_head = game_state.players[player_index].head

        start = [bot_head[0] - 5, bot_head[1] - 5]
        counter = 0

        scope = 10
        max = 0

        grid = []

        for i in range(scope):
            grid.append([])
            for j in range(scope):

                x = start[0] + i
                y = start[1] + j

                increment = self.distance_score(abs(i - 5), abs(j - 5))
                in_bounds = game_state.in_bounds([x, y])

                if in_bounds:
                    content = game_state.collision_table[x][y]
                    grid[i].append(content)

                is_bad_square = not in_bounds or content != 0

                if not in_bounds:
                    increment /= 5
                if is_bad_square:
                    counter += increment

                max += increment

        return counter, max, grid

    def distance_score(self, x, y):

        l2_distance = (x ** 2 + y ** 2) ** .5

        sqrt_50 = 50 ** .5

        distance_score = (sqrt_50 - l2_distance) / sqrt_50
        return distance_score

    def get_distances(self, game_state, player_index, possible_directions):

        head = game_state.get_heads()[player_index]
        counters = []

        for direction in possible_directions:
            x_change = game_state.DIRECTIONS[direction.value][0]
            y_change = game_state.DIRECTIONS[direction.value][1]
            x = head[0] + x_change
            y = head[1] + y_change
            counter = 0

            while game_state.in_bounds([x, y]) and game_state.collision_table[x][y] == 0:
                counter += 1
                x += x_change
                y += y_change

            counters.append(counter)
        return counters

    def print_scores(self, moves, possible_directions):
        for i in range(len(moves)):
            print("Move: ", possible_directions[i])
            print("Density : ", moves[i][0].density, "sub score: ", moves[i][0].density_subscore)
            print("Distance subscore: ", moves[i][0].distance_subscore)
            print(moves[i][0].grid)
            print("Path target : ", moves[i][0].self_evaluation.target_square)
            print("Path Subscore: ", moves[i][0].self_evaluation.path_subscore)

    def choose_direction(self, possible_directions, open_lane, player_index):

        current_direction = self.game.players[player_index].direction
        if current_direction in possible_directions:

            if random.randrange(0, 10) > 7:
                return Directions(open_lane)
            else:
                return Directions(current_direction)

        else:
            return Directions(open_lane)

