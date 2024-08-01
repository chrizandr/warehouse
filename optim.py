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


class Parameters:
    def __init__(self, dataset) -> None:
        self.num_aisles = 15
        self.num_per_row = 15

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
        # self.b = self.compute_b(cache="vars/b.nc")
        # self.pi = self.compute_pi(cache="vars/pi.nc")
        # self.mu = self.compute_mu(cache="vars/mu.nc")
        self.b = self.compute_b()
        self.pi = self.compute_pi()
        self.mu = self.compute_mu()

    def compute_pi(self, cache=None):
        logger.debug("Computing [pi]")

        if cache and os.path.exists(cache):
            logger.debug("[Using cache]")
            pi = xr.open_dataarray(cache)
            return pi

        pi = xr.DataArray(
            np.nan * np.ones((len(self.M), len(self.T))),
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

        mu = xr.DataArray(
            np.nan * np.ones((len(self.M), len(self.T_star))),
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

        b = xr.DataArray(
            np.zeros((len(self.M), len(self.T_star), len(self.T_dash))),
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


class SRRP:
    def __init__(self, dataset, router=LG_Routing) -> None:
        self.params = Parameters(dataset)
        self.router = router(
            num_aisles=self.params.num_aisles,
            num_per_row=self.params.num_per_row,
            distance_matrix=dataset.distances
        )

        self.alpha, self.beta = self.apriori_route()
        objective, vars = self.objective()
        constraints, intermediate_vars = self.constraints(vars)
        self.solve(objective, constraints, vars, intermediate_vars)

    def apriori_route(self):
        logger.debug("Computing [alpha] and [beta]")

        logger.debug("Computing the priori route")

        demand_slots = set()
        for t in self.params.T:
            non_zero_slots = self.params.r.loc[::, t]
            indices = (non_zero_slots != 0).values.nonzero()[0] + 1
            demand_slots = demand_slots.union(set(indices.tolist()))
        demand_slots = list(demand_slots)

        min_cost, apriori_route = self.router.optimize(
            [x-1 for x in demand_slots])
        apriori_route = [0] + [x+1 for x in apriori_route]
        alpha = {}
        beta = {}
        for i, slot in enumerate(tqdm(apriori_route)):
            alpha[slot] = [x for x in apriori_route[0:i]]
            beta[slot] = [x for x in apriori_route[i::]]

        return alpha, beta

    def objective(self):
        logger.debug("Constructing [Objective]")
        # Define the predefined sets
        shape_w = (len(self.params.M), len(self.params.T_star))
        shape_y = (len(self.params.M_dash), len(self.params.M_dash))
        shape_z = (len(self.params.M_dash), len(self.params.T))

        w, y = {}, {}
        for t in self.params.T_dash:
            w[t] = cp.Variable(shape_w, boolean=True)

        w_mask = np.zeros((len(self.params.T_dash), shape_w[0], shape_w[1]))
        for t in tqdm(self.params.T):
            for i in self.params.M:
                k_s = [x for x in range(int(self.params.pi.loc[i, t].item()), t)]
                w_mask[t-1, i-1, k_s] = 1

        for t in self.params.T:
            y[t] = cp.Variable(shape_y, boolean=True)

        z = cp.Variable(shape_z, boolean=True)

        objective_sum = []
        for t in self.params.T:
            objective_sum.append(self.params.t.values * y[t])

        objective = cp.Minimize(cp.sum(sum(objective_sum)))

        logger.debug("[Objective] constructed")
        return objective, (w, y, z, w_mask)

    def constraints(self, objective_variables):
        logger.debug("Constructing [Constraints]")
        constraints = []
        c, v = self.inventory_constraints(objective_variables)
        logger.debug(f"Added {len(c)} new constraints")
        constraints += c
        c, v = self.w_constraints(objective_variables, v)
        logger.debug(f"Added {len(c)} new constraints")
        constraints += c
        c, v = self.z_constraints(objective_variables, v)
        logger.debug(f"Added {len(c)} new constraints")
        constraints += c
        c, v = self.warehouse_constraints(objective_variables, v)
        logger.debug(f"Added {len(c)} new constraints")
        constraints += c

        logger.debug("[Constraints] Constructed")
        return constraints, v

    def inventory_constraints(self, objective_variables):
        logger.debug(f"Building [inventory] constraints")
        w, y, z, w_mask = objective_variables
        constraints = []

        I_dash = {1: self.params.I_dash_init.values[:, None]}
        I = {1: self.params.I_init.values[:, None]}

        # constraint (22) and (23) and (24)
        logger.debug(f"Constraint [22] and [23] and [24]")
        for t in tqdm(self.params.T):
            S = cp.multiply(w[t], self.params.b.loc[:, :, t].values)

            summations = []
            for i in self.params.M:
                indices = range(int(self.params.pi.loc[i, t].item()), t)
                summations.append(cp.sum(S[i-1, indices]))

            s = cp.vstack(summations)

            # constraint (22)
            I_dash[t+1] = I_dash[t] + self.params.p.loc[:, t].values[:, None] - s
            # constraint (23)
            constraints.append(I_dash[t] >= s)
            # constraint (24)
            I[t+1] = I[t] + s - self.params.r.loc[:, t].values[:, None]

        # constraint (37)
        logger.debug(f"Constraint [37]")
        for t in self.params.T_dash:
            if t >= 2:
                constraints.append(I[t] >= 0)
                constraints.append(I_dash[t] >= 0)

        return constraints, (I, I_dash)

    def w_constraints(self, objective_variables, intermediate_variables):
        logger.debug(f"Building constraints on [w]")
        w, y, z, w_mask = objective_variables
        I, I_dash = intermediate_variables
        constraints = []

        # constraint (25)
        logger.debug(f"Constraint [25]")
        summations = []
        for i in tqdm(self.params.M):
            k_range = range(1, int(self.params.mu.loc[i, 0].item()) + 1)
            summations.append(cp.sum([w[k][i - 1, 0] for k in k_range]))

        constraints += [cp.sum(summations) == 1]

        # constraint (26)
        logger.debug(f"Constraint [26]")
        for t in tqdm(self.params.T):
            a_s, b_s = [], []
            for i in self.params.M:
                k_range_1 = range(t+1, int(self.params.mu.loc[i, t].item()) + 1)
                k_range_2 = range(int(self.params.pi.loc[i, t].item()), t)

                a = cp.sum([w[k][i-1, t] for k in k_range_1])
                b = cp.sum([w[t][i-1, k] for k in k_range_2])
                a_s.append(a); b_s.append(b)

            A, B = cp.vstack(a_s), cp.vstack(b_s)
            constraints += [A - B == 0]

        # constraint (27)
        logger.debug(f"Constraint [27]")
        w_sums = []
        for i in tqdm(self.params.M):
            k_range = range(int(self.params.pi.loc[i, self.params.T[-1]].item()), self.params.T[-1]+1)
            w_sum = cp.sum([w[self.params.T_dash[-1]][i-1, k] for k in k_range])
            w_sums.append(w_sum)
        W = cp.vstack(w_sums)
        constraints += [W == 1]


        # constraint (28)
        logger.debug(f"Constraint [28]")
        for t in tqdm(self.params.T):
            w_s, z_s = [], []
            for i in self.params.M:
                k_range = range(int(self.params.pi.loc[i, t].item()), t)
                w_s.append(cp.sum([w[t][i-1, k] for k in k_range]))
                z_s.append(z[i, t-1])
            W = cp.vstack(w_s)
            Z = cp.vstack(z_s)
            constraints += [W == Z]

        return constraints, (I, I_dash)

    def z_constraints(self, objective_variables, intermediate_variables):
        logger.debug(f"Building constraints on [z]")
        w, y, z, w_mask = objective_variables
        I, I_dash = intermediate_variables
        constraints = []

        # constraint (29)
        logger.debug(f"Constraint [29]")
        for t in tqdm(self.params.T):
            z_s = []
            for i in self.params.M:
                z_s.append(z[i, t-1])
            constraints += [cp.vstack(z_s) <= z[0, t-1]]

        # constraint (30)
        logger.debug(f"Constraint [30]")
        for t in tqdm(self.params.T):
            y_s = []
            z_s = []
            for i in self.params.M_dash:
                if i in self.beta:
                    y_s.append(cp.sum([y[t][i, j] for j in self.beta[i]]))
                    z_s.append(z[i, t - 1])

            if len(y_s) > 0:
                Y = cp.vstack(y_s)
                Z = cp.vstack(z_s)
                constraints += [Y == Z]

        # constraint (31)
        logger.debug(f"Constraint [31]")
        for t in tqdm(self.params.T):
            y_s = []
            z_s = []
            for i in self.params.M_dash:
                if i in self.alpha:
                    y_s.append(cp.sum([y[t][j, i] for j in self.alpha[i]]))
                    z_s.append(z[i, t - 1])

            if len(y_s) > 0:
                Y = cp.vstack(y_s)
                Z = cp.vstack(z_s)
                constraints += [Y == Z]

        return constraints, (I, I_dash)

    def warehouse_constraints(self, objective_variables, intermediate_variables):
        logger.debug(f"Building constraints on [warehouse limits]")
        w, y, z, w_mask = objective_variables
        I, I_dash = intermediate_variables
        constraints = []

        # constraint (32)
        logger.debug(f"Constraint [32]")
        for t in tqdm(self.params.T):
            sum_s = []
            for i in self.params.M_dash:
                sum_s.append(cp.sum(self.params.t[i, ::].values * y[t][i, ::]))
            constraints += [cp.sum(sum_s) <= self.params.L]

        # constraint (33)
        logger.debug(f"Constraint [33]")
        for t in tqdm(self.params.T):
            sum_s = []
            for i in self.params.M:
                k_range = [x for x in range(int(self.params.pi.loc[i, t].item()), t)]
                sum_s.append(cp.sum(self.params.b.loc[i, k_range, t].values * w[t][i-1, k_range]))
            constraints += [cp.sum(sum_s) <= self.params.Q]
        return constraints, (I, I_dash)

    def solve(self, objective, constraints, vars, intermediate_vars):
        w, y, z, w_mask = vars
        I, I_dash = intermediate_vars
        # breakpoint()
        logger.debug(f"Total number of constraints: {len(constraints)}")
        logger.debug("Beginning optimisation")

        problem = cp.Problem(objective, constraints)
        solver = cp.GLPK_MI  # Change this to cp.CBC or cp.GLPK_MI as needed
        problem.solve(solver=solver, verbose=True)
        breakpoint()


if __name__ == "__main__":

    # file_path = "/home/ubuntu/warehouse/data/Large instances/25 items per cycle, 20-40 demand/450-15_inst0001.txt"
    file_path = "/home/ubuntu/warehouse/data/Small instances/i25-t03-01.dat"
    dataset = Data(file_path)
    # tsp = LG_Routing(
    #     num_aisles=num_aisles,
    #     num_per_row=num_per_row,
    #     distance_matrix=dataset.distances
    # )
    optim = SRRP(dataset)