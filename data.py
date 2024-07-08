import string
import numpy as np


class Data:
    def __init__(self, data_file) -> None:
        self.data = open(data_file).read().split("\n")
        self.load_data()

    def load_data(self):
        self.num_items = int(self.data[0].split(":")[-1].strip())
        self.num_cycles =  int(self.data[1].split(":")[-1].strip())
        self.time_limit =  int(self.data[2].split(":")[-1].strip())
        self.load_limit =  int(self.data[3].split(":")[-1].strip())
        print(f"Num items: {self.num_items}, Num cycles: {self.num_cycles}, TL: {self.time_limit}, LL: {self.load_limit}")

        counter = 4

        counter += 1    # Skip text
        self.inventory_data = np.array([int(x) for x in self.data[counter: counter+self.num_items]])
        counter += self.num_items

        counter += 1    # Skip text
        self.capacity_data = np.array([int(x) for x in self.data[counter: counter+self.num_items]])
        counter += self.num_items

        counter += 1    # Skip text
        self.reserve_cycle = np.array([[int(x) for x in text.split()]
                                  for text in self.data[counter: counter+self.num_items]])
        counter += self.num_items

        counter += 1    # Skip text
        self.forward_cycle = np.array([[int(x) for x in text.split()]
                                  for text in self.data[counter: counter+self.num_items]])
        counter += self.num_items

        counter += 1    # Skip text
        self.distances = np.array([[float(x) for x in text.split()]
                              for text in self.data[counter: counter+self.num_items]])
        self.distance_to_depot = self.distances[::, -1]
        self.distances = self.distances[::, 0:-1]

    def __len__(self):
        return self.num_items


if __name__ == "__main__":
    data_file = "/data/chris/warehouse/data/Large instances/25 items per cycle, 20-40 demand/450-15_inst0001.txt"
    dataset = Data(data_file)
    breakpoint()