import itertools

def tsp_fixed_cities(distance_matrix, cities_to_visit):
    """
    Solves the TSP for a fixed set of cities using dynamic programming.

    Parameters:
    distance_matrix: 2D list or array where distance_matrix[i][j] is the distance from city i to city j
    cities_to_visit: List of indices of the cities to visit

    Returns:
    min_cost: The minimum cost to visit all the cities in the set
    tour: The order of cities to visit for the minimum cost
    """
    n = len(distance_matrix)
    m = len(cities_to_visit)
    dp = {}

    # Initialize DP table
    for i in cities_to_visit:
        dp[(frozenset([i]), i)] = (0, [i])

    # Iterate over subsets of increasing length
    for subset_size in range(2, m+1):
        for subset in itertools.combinations(cities_to_visit, subset_size):
            subset = frozenset(subset)
            for end in subset:
                min_cost, min_path = float('inf'), []
                for k in subset:
                    if k == end:
                        continue
                    prev_subset = subset - frozenset([end])
                    cost, path = dp[(prev_subset, k)]
                    new_cost = cost + distance_matrix[k][end]
                    if new_cost < min_cost:
                        min_cost = new_cost
                        min_path = path + [end]
                dp[(subset, end)] = (min_cost, min_path)

    # Find the minimum cost to visit all cities and return to the starting city
    min_cost, min_tour = float('inf'), []
    full_set = frozenset(cities_to_visit)
    for k in cities_to_visit:
        cost, path = dp[(full_set, k)]
        if cost < min_cost:
            min_cost = cost
            min_tour = path

    return min_cost, min_tour

# Example usage:
distance_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
cities_to_visit = [0, 2, 3]

min_cost, tour = tsp_fixed_cities(distance_matrix, cities_to_visit)
print(f"Minimum cost: {min_cost}")
print(f"Tour: {tour}")
