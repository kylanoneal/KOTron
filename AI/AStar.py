from collections import defaultdict
from game.KyTron import*

def reconstruct_path(cameFrom, current):
    total_path = [current]
    while current in cameFrom:
        current = cameFrom[current]
        total_path.insert(0,current)
    return total_path


# A* finds a path from start to goal.
# h is the heuristic function. h(n) estimates the cost to reach goal from node n.
def does_path_exist(start,goal,collision_table):
    result = A_Star(start,goal,collision_table)
    if result is not None:
        return True, len(result)
    else:
        return False, 0

def A_Star(start, goal, collision_table):
    # The set of discovered nodes that may need to be (re-)expanded.
    # Initially, only the start node is known.
    # This is usually implemented as a min-heap or priority queue rather than a hash-set.
    openSet = [start]

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start
    # to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = defaultdict(lambda: 420000)
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n). fScore[n] represents our current best guess as to
    # how short a path from start to finish can be if it goes through n.
    fScore = defaultdict(lambda: 420000)
    fScore[start] = get_l1_distance(start, goal)

    while len(openSet) != 0:
        # This operation can occur in O(1) time if openSet is a min-heap or a priority queue
        open_set_scores = []
        for node in openSet:
            open_set_scores.append(fScore[node])

        current_index = open_set_scores.index(min(open_set_scores))
        current = openSet[current_index]


        if current == goal:
            return reconstruct_path(cameFrom, current)

        openSet.pop(current_index)

        for neighbor in get_neighbors(current, collision_table):
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore = gScore[current] + 1

            if tentative_gScore < gScore[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + get_l1_distance(neighbor, goal)
                if neighbor not in openSet:
                    openSet.append(neighbor)

    # Open set is empty but goal was never reached
    return None


def get_l1_distance(start, goal):
    return abs(goal[0] - start[0]) + abs(goal[1] - start[1])

def get_neighbors(coordinate, collision_table):

    neighbors = []

    for i in range(4):


        x = coordinate[0] + BMTron.DIRECTIONS[i][0]
        y = coordinate[1] + BMTron.DIRECTIONS[i][1]

        if in_bounds([x,y], len(collision_table)):

            if collision_table[x][y] == 0:

                neighbors.append((x,y))

    return neighbors

def in_bounds(coords, dimension):
    return 0 <= coords[0] < dimension and 0 <= coords[1] < dimension

def main():
    wall_test_case()


def wall_test_case():
    collision_table = BMTron.build_collision_table(50)

    for i in range(50):
        if i != 25:

            collision_table[16][i] = 1


    path = A_Star((0, 0), (25, 25), collision_table)
    print("path: ", path)

def simple_test_case():
    path = A_Star((0,0), (25, 25), BMTron.build_collision_table(50))
    print("path: ", path)

if __name__ == '__main__':
    main()