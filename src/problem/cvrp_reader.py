import re
import numpy as np


class CVRPInstance:
    """Represents a CVRP problem instance"""
    
    def __init__(self, name, dimension, capacity, depot, customers):
        self.name = name
        self.dimension = dimension
        self.capacity = capacity
        self.depot = depot  # (x, y) coordinates
        self.customers = customers  # dict: id -> {'coords': (x,y), 'demand': d}
        
    def __str__(self):
        return f"CVRP({self.name}, {self.dimension} customers, capacity={self.capacity})"


def read_cvrp_file(filepath):
    """
    Parse CVRPLIB format file
    
    Returns:
        CVRPInstance object
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    name = None
    dimension = None
    capacity = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('NAME'):
            name = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            capacity = int(line.split(':')[1].strip())
    
    # Parse coordinates
    coords = {}
    in_coord_section = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('NODE_COORD_SECTION'):
            in_coord_section = True
            continue
        elif line.startswith('DEMAND_SECTION'):
            in_coord_section = False
            break
        
        if in_coord_section and line and not line.startswith('EOF'):
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[node_id] = (x, y)
    
    # Parse demands
    demands = {}
    in_demand_section = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('DEMAND_SECTION'):
            in_demand_section = True
            continue
        elif line.startswith('DEPOT_SECTION'):
            in_demand_section = False
            break
        
        if in_demand_section and line and not line.startswith('EOF'):
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                demand = int(parts[1])
                demands[node_id] = demand
    
    # Depot is node 1 with demand 0
    depot_coords = coords[1]
    
    # Customers are nodes 2 to dimension
    customers = {}
    for i in range(2, dimension + 1):
        if i in coords and i in demands:
            customers[i] = {
                'coords': coords[i],
                'demand': demands[i]
            }
    
    return CVRPInstance(name, dimension, capacity, depot_coords, customers)


if __name__ == '__main__':
    # Test the reader
    instance = read_cvrp_file('../data/A-n32-k5.vrp')
    print(instance)
    print(f"Depot: {instance.depot}")
    print(f"First customer: {instance.customers[2]}")
    print(f"Total customers: {len(instance.customers)}")