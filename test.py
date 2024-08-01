import cvxpy as cp
import numpy as np

# Define the dimensions of the matrix
rows, cols = 3, 3

# Define the variable matrix
X = cp.Variable((rows, cols))

# Define the mask for constants (1 where the value should be constant, 0 where it should be a variable)
mask = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

# Define the matrix of constants
constants = np.array([
    [5, 0, 0],
    [0, 8, 0],
    [0, 0, 0]
])

# Define the combined matrix with constants fixed
fixed_X = cp.multiply(mask, constants) + cp.multiply(1 - mask, X)

# Example objective function (minimize the sum of all elements)
objective = cp.Minimize(cp.sum(fixed_X))

# Example constraints (all elements should be non-negative)
constraints = [fixed_X >= 0]

# Define the problem
problem = cp.Problem(objective, constraints)

solver = cp.CBC  # Change this to cp.CBC or cp.GUROBI as needed
# Solve the problem
problem.solve(solver=solver, verbose=True)

# Print the optimized variable matrix
print("Optimized variable matrix:")
print(X.value)

# Print the final matrix with constants fixed and variables optimized
final_matrix = mask * constants + (1 - mask) * X.value
print("Final matrix with constants fixed:")
print(final_matrix)
