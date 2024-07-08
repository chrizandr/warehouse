import numpy as np
from tqdm import tqdm
from data import Data
import itertools


class TSP_DP:
    def __init__(self, distance_matrix=None, num_aisles=None, num_per_aisle=None, cost_in_aisle=None,
                 cost_across_aisle=None, cost_adjacent_slot=None) -> None:
        if distance_matrix is None:
            assert num_aisles is not None
            assert num_per_aisle is not None
            assert cost_in_aisle is not None
            assert cost_across_aisle is not None
            assert cost_adjacent_slot is not None
            print("No distance matrix provided, building basic warehouse layout")
            # self.distances = self.build_basic_layout(
            #     num_slots, num_per_aisle, cost_in_aisle,
            #     cost_across_aisle, cost_adjacent_slot
            # )
            self.distances = self.generate_distance_matrix(
                num_aisles, num_per_aisle, cost_adjacent_slot,
                cost_in_aisle,
                cost_across_aisle,
            )
        else:
            self.distances = distance_matrix

    def tsp_base(self, cities):

        D = self.distances[np.ix_(cities, cities)]
        padded_size = D.shape[0] + 1
        dist = np.zeros((padded_size, padded_size))
        # Place the original subset matrix in the bottom right corner of the padded matrix
        dist[1:, 1:] = D

        n = len(cities)
        memo = np.array([[-1]*(1 << (n+1)) for _ in range(n+1)])

        def fun(i, mask):
            # base case
            # if only ith bit and 1st bit is set in our mask,
            # it implies we have visited all other nodes already
            if mask == ((1 << i) | 3):
                return dist[1][i]

            # memoization
            if memo[i][mask] != -1:
                return memo[i][mask]

            res = 10**9  # result of this sub-problem

            # we have to travel all nodes j in mask and end the path at ith node
            # so for every node j in mask, recursively calculate cost of
            # travelling all nodes in mask
            # except i and then travel back from node j to node i taking
            # the shortest path take the minimum of all possible j nodes
            for j in range(1, n+1):
                if (mask & (1 << j)) != 0 and j != i and j != 1:
                    res = min(res, fun(j, mask & (~(1 << i))) + dist[j][i])
            memo[i][mask] = res  # storing the minimum value
            return res


        # Driver program to test above logic
        ans = 10**9
        for i in range(1, n+1):
            # try to go from node 1 visiting all nodes in between to i
            # then return from i taking the shortest route to 1
            ans = min(ans, fun(i, (1 << (n+1))-1) + dist[i][1])

        print("The cost of most efficient tour = " + str(ans))


    def tsp_subset(self, cities_to_visit):
        """
        Solves the TSP for a fixed set of cities using dynamic programming,
        ensuring the tour starts and ends at the first city in cities_to_visit.

        Parameters:
        distance_matrix: 2D list or array where distance_matrix[i][j] is the distance from city i to city j
        cities_to_visit: List of indices of the cities to visit, starting and ending at the first city

        Returns:
        min_cost: The minimum cost to visit all the cities in the set and return to the start
        tour: The order of cities to visit for the minimum cost
        """
        distance_matrix = self.distances
        n = len(distance_matrix)
        m = len(cities_to_visit)
        start_city = cities_to_visit[0]
        num_subsets = 1 << m  # 2^m subsets

        # Initialize DP table
        dp = [[float('inf')] * m for _ in range(num_subsets)]
        parent = [[-1] * m for _ in range(num_subsets)]
        dp[1 << 0][0] = 0  # Start at the first city

        # Populate DP table
        for mask in range(num_subsets):
            for i in range(m):
                if mask & (1 << i):  # If city i is in the subset
                    for j in range(m):
                        if mask & (1 << j) == 0:  # If city j is not in the subset
                            next_mask = mask | (1 << j)
                            new_cost = dp[mask][i] + distance_matrix[cities_to_visit[i]][cities_to_visit[j]]
                            if new_cost < dp[next_mask][j]:
                                dp[next_mask][j] = new_cost
                                parent[next_mask][j] = i

        # Find minimum cost to return to the start city
        min_cost = float('inf')
        last_city = -1
        final_mask = (1 << m) - 1  # All cities visited
        for i in range(1, m):
            cost = dp[final_mask][i] + distance_matrix[cities_to_visit[i]][start_city]
            if cost < min_cost:
                min_cost = cost
                last_city = i

        # Reconstruct the tour
        tour = [start_city]
        mask = final_mask
        while last_city != -1:
            tour.append(cities_to_visit[last_city])
            next_last_city = parent[mask][last_city]
            mask ^= (1 << last_city)
            last_city = next_last_city
        tour.append(start_city)
        tour.reverse()

        return min_cost, tour

    def build_basic_layout(self,
                num_slots, num_per_aisle, cost_adjacent_slot,
                cost_in_aisle,
                cost_across_aisle,
            ):

        aisle_row = num_slots // num_per_aisle // 2
        num_aisles = num_slots // (aisle_row * 2)

        total_vertices = 0
        all_vertices = []
        connections = np.inf * np.ones((num_slots, num_slots))

        for i in range(num_aisles):
            row_indices = total_vertices + np.arange(aisle_row)
            for d in range(aisle_row):
                j = row_indices[d]
                connections[j][j] = connections[j+aisle_row][j+aisle_row] = 0
                connections[j][j+aisle_row] = cost_in_aisle
                connections[j+aisle_row][j] = cost_in_aisle
                if d != aisle_row - 1:
                    connections[j][j+1] = connections[j+1][j] = cost_adjacent_slot
                    connections[j+aisle_row][j+aisle_row+1] = connections[j+aisle_row+1][j+aisle_row] = cost_adjacent_slot

            total_vertices += aisle_row * 2
            if total_vertices != num_slots:
                connections[total_vertices-1][total_vertices] = connections[total_vertices][total_vertices-1] = cost_across_aisle
        for i in range(connections.shape[0]):
            for j in range(connections.shape[1]):
                if j % aisle_row == 0:
                    continue
                connections[i, j] = min(connections[i-1, j], connections[i, j], connections[i,j], )

        return connections

    def generate_distance_matrix(self, n_aisles, m_shelves, d_s, d_r, d_a):
        total_shelves = 2 * n_aisles * m_shelves
        distance_matrix = np.inf * np.ones((total_shelves, total_shelves))

        def shelf_position(shelf):
            aisle = shelf // (2 * m_shelves)
            row = (shelf % (2 * m_shelves)) // m_shelves
            pos_in_row = shelf % m_shelves
            return aisle, row, pos_in_row

        for i in tqdm(range(total_shelves)):
            for j in range(total_shelves):
                if i == j:
                    distance_matrix[i][j] = 0
                    continue

                aisle_i, row_i, pos_i = shelf_position(i)
                aisle_j, row_j, pos_j = shelf_position(j)

                if aisle_i == aisle_j:
                    if row_i == row_j:
                        distance = abs(pos_i - pos_j) * d_s
                    else:
                        distance = abs(pos_i - pos_j) * d_s + d_r
                else:
                    # Calculate distance to exit the aisle from top or bottom
                    distance_to_exit_i = min(pos_i * d_s, (m_shelves - pos_i - 1) * d_s)
                    distance_to_exit_j = min(pos_j * d_s, (m_shelves - pos_j - 1) * d_s)

                    # Total distance includes exiting both aisles and moving between aisles
                    distance = distance_to_exit_i + distance_to_exit_j + abs(aisle_i - aisle_j) * d_a

                distance_matrix[i][j] = distance

        return distance_matrix


if __name__ == "__main__":
    num_aisles = 15
    num_per_aisle = 15
    cost_in_aisle = 0
    cost_across_aisle = 1
    cost_adjacent_slot = 1
    data_file = "/data/chris/warehouse/data/Large instances/25 items per cycle, 20-40 demand/450-15_inst0001.txt"

    dataset = Data(data_file)

    tsp = TSP_DP(
        num_aisles=num_aisles,
        num_per_aisle=num_per_aisle,
        cost_in_aisle=cost_in_aisle,
        cost_across_aisle=cost_across_aisle,
        cost_adjacent_slot=cost_adjacent_slot,
    )
    min_cost, min_tour = tsp.tsp_base([
        0, 8, 14, 16, 27, 38, 59, 90, 126, 143, 169, 187, 195,
        204, 217, 226, 235
    ])
    breakpoint()