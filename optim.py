import logging
import os
import warnings

import cvxpy as cp
import numpy as np
from tqdm import tqdm
import xarray as xr

from data import Data
from largest_gap import LG_Routing

# Suppress all warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler = logging.FileHandler('output.log')
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)



class SRRP:
    def __init__(self, dataset, router=LG_Routing) -> None:

        self.M = list(range(1, dataset.num_items + 1))
        self.M_dash = [0] + self.M
        self.T = list(range(1, dataset.num_cycles+1))

        self.t = np.pad(dataset.distances, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        self.t[1::,0] = self.t[0,1::] = dataset.distance_to_depot
        self.t = xr.DataArray(self.t, coords=[self.M_dash, self.M_dash], dims=['i', 'j'])

        self.p = xr.DataArray(dataset.reserve_cycle, coords=[self.M, self.T], dims=["i", "t"])
        self.r = xr.DataArray(dataset.forward_cycle, coords=[self.M, self.T], dims=["i", "t"])

        self.U = xr.DataArray(dataset.capacity_data, coords=[self.M], dims=["i"])

        self.L = dataset.time_limit
        self.Q = dataset.load_limit

        self.I_init = xr.DataArray(dataset.inventory_data, coords=[self.M], dims=["i"])
        self.I_dash_init = xr.DataArray(self.r[::, 0], coords=[self.M], dims=["i"])

        self.T_star = [0] + self.T
        self.T_dash = self.T + [len(self.T)+1]

        logger.debug("Computing parameters...")
        self.b = self.compute_b(cache="vars/b.nc")
        self.pi = self.compute_pi(cache="vars/pi.nc")
        self.mu = self.compute_mu(cache="vars/mu.nc")
        objective, vars = self.objective()
        constraints = self.constraints(vars)
        breakpoint()

        self.process_cycle()


    def compute_pi(self, cache=None):
        logger.debug("Computing [pi]")

        if cache and os.path.exists(cache):
            logger.debug("[Using cache]")
            pi = xr.open_dataarray(cache)
            return pi

        pi = xr.DataArray(np.nan * np.ones((len(self.M), len(self.T))),
                          coords=[self.M, self.T],
                          dims=["i", "t"])
        for i in tqdm(self.M):
            for t in self.T:
                Ui_val = self.U.loc[i].item()
                bikt_slice = self.b.loc[i, : , t]
                mask_val = bikt_slice <= Ui_val
                mask_range = (self.b.coords['k'] >= 0) & (self.b.coords['k'] <= t-1)
                mask = mask_range & mask_val

                filled_bikt = xr.where(mask, bikt_slice, np.inf)

                min_k_index = int(filled_bikt.idxmin(dim='k').item())
                if np.isfinite(filled_bikt.sel(k=min_k_index).item()):
                    pi.loc[{'i': i, 't': t}] = min_k_index
                else:
                    logger.error("Demand is greater than capacity.....terminating optimization")
                    raise ValueError("Demand is greater than capacity")
        if cache:
            pi.to_netcdf(cache)

        return pi

    def compute_mu(self, cache=None):
        logger.debug("Computing [mu]")

        if cache and os.path.exists(cache):
            logger.debug("[Using cache]")
            mu = xr.open_dataarray(cache)
            return mu

        mu = xr.DataArray(np.nan * np.ones((len(self.M), len(self.T_star))),
                          coords=[self.M, self.T_star],
                          dims=["i", "t"])
        for i in tqdm(self.M):
            for t in self.T_star:
                Ui_val = self.U.loc[i].item()
                bitk = self.b.loc[i, t, :]
                mask_val = bitk <= Ui_val
                mask_range = (self.b.coords['t'] >= t+1) & (self.b.coords['t'] <= self.T_dash[-1])
                mask = mask_range & mask_val

                filled_bikt = xr.where(mask, bitk, -1 * np.inf)

                max_k_index = int(filled_bikt.idxmax(dim='t').item())

                if np.isfinite(filled_bikt.sel(t=max_k_index).item()):
                    mu.loc[{'i': i, 't': t}] = max_k_index
                else:
                    logger.error("Demand is greater than capacity.....terminating optimization")
                    raise ValueError("Demand is greater than capacity")

        if cache:
            mu.to_netcdf(cache)
        return mu

    def compute_b(self, cache=None):
        logger.debug("Computing [b]")

        if cache and os.path.exists(cache):
            logger.debug("[Using cache]")
            b = xr.open_dataarray(cache)
            return b

        b = xr.DataArray(np.zeros((len(self.M), len(self.T_star), len(self.T_dash))),
                         coords=[self.M, self.T_star, self.T_dash],
                         dims=["i", "k", "t"])

        for t in self.T_dash:
            indices = list(range(1, t))
            b.loc[:, 0, t] = self.U.loc[:] - self.I_init.loc[:] + self.r.loc[:, indices].sum(axis=1)

        for k in self.T:
            for t in self.T_dash:
                indices = list(range(k, t))
                b.loc[:, k, t] = self.r.loc[:, indices].sum(axis=1)

        if cache:
            b.to_netcdf(cache)

        return b

    def process_cycle(self, apriori_route):
        for i in range(len(self.T)):
            self.apriori_route = apriori_route
            self.aplha, self.beta = self.process_route(apriori_route)


    def process_route(self, route):
        alpha = {}
        beta = {}
        for i, slot in enumerate(route):
            alpha[slot] = [0] + [x + 1 for x in route[0:i]]
            beta[slot] = [x + 1 for x in route[i::]] + [0]
        return alpha, beta

    def objective(self):
        logger.debug("Constructing [Objective]")
        # Define the predefined sets
        shape_w = (len(self.M), len(self.T_star))
        shape_y = (len(self.M_dash), len(self.M_dash))
        shape_z = (len(self.M_dash), len(self.T))

        w, y = {}, {}
        for t in self.T_dash:
            w[t] = cp.Variable(shape_w, boolean=True)

        for t in self.T:
            y[t] = cp.Variable(shape_y, boolean=True)

        z = cp.Variable(shape_z, boolean=True)

        objective_sum = []
        for t in self.T:
            objective_sum.append(self.t.values * y[t])

        objective = cp.Minimize(cp.sum(sum(objective_sum)))

        logger.debug("[Objective] constructed")
        return objective, (w, y, z)

    def constraints(self, objective_variables):
        logger.debug("Constructing [Constraints]")
        constraints = []
        c, v = self.constraints_22_to_24_and_37(objective_variables)
        logger.debug(f"Added {len(c)} new constraints")
        constraints += c
        c, v = self.constraints_25_to_28(objective_variables, v)
        logger.debug(f"Added {len(c)} new constraints")
        constraints += c

    def constraints_22_to_24_and_37(self, objective_variables):
        w, y, z = objective_variables
        constraints = []

        I_dash = {1: self.I_dash_init.values[:, None]}
        I = {1: self.I_init.values[:, None]}


        for t in tqdm(self.T):
            S = cp.multiply(w[t], self.b.loc[:, :, t].values)

            summations = []
            for i in self.M:
                indices = range(int(self.pi.loc[i, t].item()), t)
                summations.append(cp.sum(S[i-1, indices]))

            s = cp.vstack(summations)

            # constraint (22)
            I_dash[t+1] = I_dash[t] + self.p.loc[:, t].values[:, None] - s
            # constraint (23)
            constraints.append(I_dash[t] >= s)
            # constraint (24)
            I[t+1] = I[t] + s - self.r.loc[:, t].values[:, None]

        for t in self.T_dash:
            if t >= 2:
                # constraint (37)
                constraints.append(I[t] >= 0)
                constraints.append(I_dash[t] >= 0)

        return constraints, (I, I_dash)

    def constraints_25_to_28(self, objective_variables, intermediate_variables):
        w, y, z = objective_variables
        I, I_dash = intermediate_variables
        constraints = []

        # constraint (25)
        summations = []
        for i in self.M:
            k_range = range(1, int(self.mu.loc[i, 0].item()) + 1)
            summations.append(cp.sum([w[k][i - 1, 0] for k in k_range]))

        constraints += [cp.sum(summations) == 1]


        # TODO: Check indices and stack the subtractions to reduce number of constraints
        # constraint (26)
        for t in tqdm(self.T):
            a_s, b_s = [], []
            for i in self.M:
                k_range_1 = range(t+1, int(self.mu.loc[i, t].item()) + 1)
                k_range_2 = range(int(self.pi.loc[i, t].item()), t)

                a = cp.sum([w[k][i-1, t] for k in k_range_1])
                b = cp.sum([w[t][i-1, k] for k in k_range_2])
                a_s.append(a); b_s.append(b)

            A, B = cp.vstack(a_s), cp.vstack(b_s)
            constraints += [A - B == 0]
        return constraints, (I, I_dash)
        breakpoint()












    def solve(self, objective, constraints, variables):
        problem = cp.Problem(objective, constraints)
        solver = cp.GLPK_MI  # Change this to cp.CBC or cp.GUROBI as needed
        problem.solve(solver=solver, verbose=True)




if __name__ == "__main__":
    num_aisles = 15
    num_per_row = 15
    file_path = "/data/chris/warehouse/data/Large instances/300 items per cycle, uniform demand/450-15_inst0001.txt"
    dataset = Data(file_path)
    tsp = LG_Routing(
        num_aisles=num_aisles,
        num_per_row=num_per_row,
        distance_matrix=dataset.distances
    )
    optim = SRRP(dataset)