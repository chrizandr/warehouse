from data import Data
from tqdm import tqdm
import numpy as np
import xarray as xr
from largest_gap import LG_Routing


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

        self.b = self.compute_b()
        self.pi = self.compute_pi()
        self.mu = self.compute_mu()

        self.process_cycle()


    def compute_pi(self):
        pi = xr.DataArray(np.nan * np.ones((len(self.M), len(self.T))),
                          coords=[self.M, self.T],
                          dims=["i", "t"])
        for i in tqdm(self.M):
            for t in self.T:
                Ui_val = self.U.sel(i=i)
                bikt_slice = self.b.sel(i=i, t=t)

                mask_val = bikt_slice <= Ui_val
                mask_range = (self.b.coords['k'] >= 0) & (self.b.coords['k'] <= t-1)
                mask = mask_range & mask_val

                filled_bikt = xr.where(mask, bikt_slice, np.inf)

                min_k_index = filled_bikt.argmin(dim='k')

                if np.isfinite(filled_bikt.isel(k=min_k_index).item()):
                    pi.loc[{'i': i, 't': t}] = min_k_index
        return pi

    def compute_mu(self):
        mu = xr.DataArray(np.nan * np.ones((len(self.M), len(self.T_dash))),
                          coords=[self.M, self.T_dash],
                          dims=["i", "t"])
        for i in tqdm(self.M):
            for t in self.T:
                Ui_val = self.U.sel(i=i)
                bikt_slice = self.b.sel(i=i, k=t)

                mask_val = bikt_slice <= Ui_val
                mask_range = (self.b.coords['t'] >= t+1) & (self.b.coords['t'] <= self.T_dash[-1])
                mask = mask_range & mask_val

                filled_bikt = xr.where(mask, bikt_slice, -1*np.inf)

                max_k_index = filled_bikt.argmax(dim='t')
                # if max_k_index.item() == 0:
                #     breakpoint()

                if np.isfinite(filled_bikt.isel(t=max_k_index).item()):
                    mu.loc[{'i': i, 't': t}] = max_k_index

        return mu


    def compute_b(self):
        b = xr.DataArray(np.zeros((len(self.M), len(self.T_star), len(self.T_dash))),
                         coords=[self.M, self.T_star, self.T_dash],
                         dims=["i", "k", "t"])

        for t in self.T_dash:
            b.loc[:, 0, t] = self.U.loc[:] - self.I_init.loc[:] + self.r.loc[:, 1:t].sum(axis=1)

        for k in self.T:
            for t in self.T_dash:
                b.loc[:, k, t] = self.r.loc[:, k:t].sum(axis=1)

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