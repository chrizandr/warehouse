import numpy as np
from tqdm import tqdm
from data import Data
import itertools


class LG_Routing:
    def __init__(self, distance_matrix=None, num_aisles=None, num_per_row=None, cost_in_aisle=None,
                 cost_across_aisle=None, cost_adjacent_slot=None) -> None:
        assert num_aisles is not None
        assert num_per_row is not None

        if distance_matrix is None:
            assert cost_in_aisle is not None
            assert cost_across_aisle is not None
            assert cost_adjacent_slot is not None
            print("No distance matrix provided, building basic warehouse layout")
            self.distances = self.generate_distance_matrix(
                num_aisles, num_per_row, cost_adjacent_slot,
                cost_in_aisle,
                cost_across_aisle,
            )
        else:
            self.distances = distance_matrix

        self.num_aisles = num_aisles
        self.num_per_row = num_per_row
        self.cost_in_aisle = cost_in_aisle
        self.cost_across_aisle = cost_across_aisle
        self.cost_adjacent_slot = cost_adjacent_slot

    @staticmethod
    def shelf_position(shelf, m_shelves):
            aisle = shelf // (2 * m_shelves)
            row = (shelf % (2 * m_shelves)) // m_shelves
            pos_in_row = shelf % m_shelves
            return aisle, row, pos_in_row

    @staticmethod
    def slot_number(aisle, row, position, m_shelves):
            pos = (m_shelves * aisle * 2) + row * m_shelves + position
            return pos

    def optimize(self, slots):
        total_shelves = 2 * self.num_aisles * self.num_per_row
        aisles = [list() for _ in range(self.num_aisles)]

        for slot in slots:
            aisle, row, pos = LG_Routing.shelf_position(slot, self.num_per_row)
            aisles[aisle].append((aisle, row, pos))

        split_aisles = [[list(), list()] for _ in range(self.num_aisles)]

        for i, aisle in enumerate(aisles):
            sorted_slots = sorted(aisle, key= lambda x:x[2])

            if len(sorted_slots) == 0:
                continue

            if i == 0:
                bottom = sorted_slots
                top = []
                split_aisles[i][0] = top
                split_aisles[i][1] = bottom
                continue

            if i == self.num_aisles - 1:
                top = sorted_slots
                bottom = []
                split_aisles[i][0] = top
                split_aisles[i][1] = bottom
                continue

            largest_gap = 0
            gap_index = (-1, -1)

            for j in range(len(sorted_slots) - 1):
                if sorted_slots[j + 1][2] - sorted_slots[j][2] > largest_gap:
                    largest_gap = sorted_slots[j + 1][2] - sorted_slots[j][2]
                    gap_index = j+1

            bottom_gap = sorted_slots[0][2]
            top_gap = self.num_per_row - sorted_slots[-1][2]

            if largest_gap > bottom_gap and largest_gap > top_gap:
                split = gap_index
            elif bottom_gap > top_gap:
                split = 0
            elif top_gap >= bottom_gap:
                split = len(sorted_slots)

            bottom = sorted_slots[0:split]
            top = sorted_slots[split::]
            split_aisles[i][0] = top
            split_aisles[i][1] = bottom

        top_route = [self.slot_number(*x, self.num_per_row) for x in split_aisles[0][1]]
        bottom_start = [self.slot_number(*x, self.num_per_row) for x in split_aisles[-1][0][::-1]]
        bottom_route = []
        for i, aisle in enumerate(split_aisles[1:-1]):
            top_route.extend(sorted([self.slot_number(*x, self.num_per_row) for x in aisle[0][::-1]]))
            bottom_route.extend(sorted([self.slot_number(*x, self.num_per_row) for x in aisle[1][::-1]]))

        route = top_route + bottom_start + bottom_route[::-1]
        est_cost = sum([self.distances[route[i], route[i+1]] for i in range(len(route) - 1)])
        return est_cost, route

    def generate_distance_matrix(self, num_aisles, m_shelves, d_s, d_r, d_a):
        total_shelves = 2 * num_aisles * m_shelves
        distance_matrix = np.inf * np.ones((total_shelves, total_shelves))

        for i in tqdm(range(total_shelves)):
            for j in range(total_shelves):
                if i == j:
                    distance_matrix[i][j] = 0
                    continue

                aisle_i, row_i, pos_i = self.shelf_position(i, m_shelves)
                aisle_j, row_j, pos_j = self.shelf_position(j, m_shelves)

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
    num_per_row = 15
    cost_in_aisle = 0
    cost_across_aisle = 1
    cost_adjacent_slot = 1
    data_file = "/data/chris/warehouse/data/Large instances/300 items per cycle, uniform demand/450-15_inst0001.txt"

    dataset = Data(data_file)

    # tsp = LG_Routing(
    #     num_aisles=num_aisles,
    #     num_per_row=num_per_row,
    #     cost_in_aisle=cost_in_aisle,
    #     cost_across_aisle=cost_across_aisle,
    #     cost_adjacent_slot=cost_adjacent_slot,
    # )

    # min_cost, min_tour = tsp.largest_gap_route(
    #         [
    #             8, 14, 16, 27, 38, 59, 90, 126, 143, 169, 187, 195,
    #             204, 217, 226, 235
    #         ],
    #         num_aisles, num_per_row, cost_adjacent_slot,
    #         cost_in_aisle,
    #         cost_across_aisle,
    #     )

    tsp = LG_Routing(
        num_aisles=num_aisles,
        num_per_row=num_per_row,
        distance_matrix=dataset.distances
    )

    routes = []
    for cycle in range(dataset.forward_cycle.shape[1]):
        cycle_demand = dataset.forward_cycle[::, cycle].nonzero()[0].tolist()
        min_cost, min_tour = tsp.optimize(cycle_demand)
        routes.append((min_cost, min_tour))
    breakpoint()