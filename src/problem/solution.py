import numpy as np


class CVRPSolution:
    """
    Represents a solution to the CVRP problem
    
    A solution is a set of routes, where each route is a list of customer IDs
    starting and ending at the depot (represented as 0)
    """

    EVAL_COUNTER = 0 # Global evaluation counter for MOEAs
    
    def __init__(self, routes, instance):
        """
        Args:
            routes: List of routes, e.g., [[0,2,5,0], [0,3,4,0]]
                    where 0 represents the depot
            instance: CVRPInstance object
        """

        CVRPSolution.EVAL_COUNTER += 1
        self.routes = routes
        self.instance = instance
        
        # Calculate objectives
        self.total_distance = self._calculate_total_distance()
        self.route_balance = self._calculate_route_balance()
        
        # For MOEA use
        self.rank = None
        self.crowding_distance = None
        self.dominated_by = []
        self.dominates_count = 0
        
    def _calculate_total_distance(self):
        """Objective 1: Total distance of all routes"""
        total = 0.0
        for route in self.routes:
            total += self._route_distance(route)
        return total
    
    def _calculate_route_balance(self):
        """Objective 2: Maximum route length (min-max fairness)"""
        if not self.routes:
            return 0.0
        
        route_lengths = [self._route_distance(r) for r in self.routes]
        return max(route_lengths)
        
        # Alternative: Standard deviation
        # return np.std(route_lengths)
    
    def _route_distance(self, route):
        """Calculate distance of a single route"""
        if len(route) <= 2:  # Only depot visits
            return 0.0
        
        dist = 0.0
        for i in range(len(route) - 1):
            dist += self._distance_between(route[i], route[i+1])
        return dist
    
    def _distance_between(self, node1, node2):
        """Euclidean distance between two nodes"""
        coord1 = self._get_coords(node1)
        coord2 = self._get_coords(node2)
        
        return np.sqrt((coord1[0] - coord2[0])**2 + 
                      (coord1[1] - coord2[1])**2)
    
    def _get_coords(self, node):
        """Get coordinates for a node (0=depot, >0=customer)"""
        if node == 0:
            return self.instance.depot
        else:
            return self.instance.customers[node]['coords']
    
    def is_feasible(self):
        """Check if solution respects capacity constraints"""
        for route in self.routes:
            load = self._route_load(route)
            if load > self.instance.capacity:
                return False
        return True
    
    def _route_load(self, route):
        """Calculate total demand for a route"""
        load = 0
        for node in route:
            if node != 0:  # Skip depot
                load += self.instance.customers[node]['demand']
        return load
    
    def get_route_info(self):
        """Get detailed route information"""
        info = []
        for i, route in enumerate(self.routes):
            info.append({
                'route_id': i + 1,
                'customers': [c for c in route if c != 0],
                'load': self._route_load(route),
                'distance': self._route_distance(route)
            })
        return info
    
    def __str__(self):
        return f"Solution(distance={self.total_distance:.2f}, " \
               f"balance={self.route_balance:.2f}, " \
               f"routes={len(self.routes)}, feasible={self.is_feasible()})"


if __name__ == '__main__':
    # Test solution representation
    from cvrp_reader import read_cvrp_file
    
    instance = read_cvrp_file('../../data/A-n32-k5.vrp')
    
    # Example solution: 2 routes
    routes = [
        [0, 2, 3, 4, 0],  # Depot -> customers 2,3,4 -> Depot
        [0, 5, 6, 0]       # Depot -> customers 5,6 -> Depot
    ]
    
    solution = CVRPSolution(routes, instance)
    print(solution)
    print(f"Feasible: {solution.is_feasible()}")
    print("\nRoute details:")
    for route_info in solution.get_route_info():
        print(route_info)