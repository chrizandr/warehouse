import numpy as np


num_aisles = 5
num_slots = 10
travel_time_between_aisles = 1
travel_time_within_aisle = 0.1

items_to_pick = np.random.choice(num_aisles * num_slots, size=10, replace=False)
print(items_to_pick)

aisles = items_to_pick // num_slots
slots = items_to_pick % num_slots


dp = np.full((num_aisles, num_slots), np.inf)
dp[0][0] = 0


for aisle in range(num_aisles):
    for slot in range(num_slots):
        if aisle > 0:
            dp[aisle][slot] = min(dp[aisle][slot], dp[aisle-1][-1] + travel_time_between_aisles)
        if slot > 0:
            dp[aisle][slot] = min(dp[aisle][slot], dp[aisle][slot-1] + travel_time_within_aisle)


optimal_route = []
current_aisle = num_aisles - 1
current_slot = num_slots - 1
while current_aisle > 0 or current_slot > 0:
    optimal_route.append((current_aisle, current_slot))
    if current_slot > 0 and dp[current_aisle][current_slot] == dp[current_aisle][current_slot-1] + travel_time_within_aisle:
        current_slot -= 1
    else:
        current_aisle -= 1
optimal_route.append((0, 0))


optimal_route = optimal_route[::-1]
print("Optimal Picking Route:", optimal_route)