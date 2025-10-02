"""
NSGA-II implementation for CVRP
"""
import random
import numpy as np
from src.algorithms.utils import fast_non_dominated_sort
from src.operators.ox import ox_crossover, mutate_swap
from src.problem.split import routes_to_tour, tour_to_routes
from src.problem.solution import CVRPSolution




class NSGA2:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II)
    
    Key features:
    - Fast non-dominated sorting
    - Crowding distance for diversity
    - Elitist selection
    """
    
    def __init__(self, instance, pop_size=100, generations=500,
                 crossover_prob=0.7, mutation_prob=0.2, seed=None):
        self.instance = instance
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.seed = seed
        
        # Get list of customer IDs
        self.customers = list(instance.customers.keys())
        
    def run(self):

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            
        print(f"Initializing population of {self.pop_size}...")
        population = self.initialize_population()

        # Initial non-dominated sorting and crowding distance
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            self.calculate_crowding_distance(front)

        for gen in range(self.generations):
            # Create offspring population
            offspring = self.create_offspring(population)

            # Combine parent and offsprinmg
            combined = population + offspring

            # Non-dominated sort on combined population
            fronts = fast_non_dominated_sort(combined)

            # Recompute crowding distance for each front
            for front in fronts:
                self.calculate_crowding_distance(front)

            # Select next generation
            population = self.select_next_generation(fronts)
            assert len(population) == self.pop_size, "Population size mismatch!"

            # Optional progress
            if (gen + 1) % 50 == 0 or gen == self.generations - 1:
                print(f"Generation {gen + 1}/{self.generations} - Front 1 size: {len(fronts[0])}")

        final_fronts = fast_non_dominated_sort(population)
        if final_fronts and len(final_fronts[0]) > 0:
            print(f"\nFinal Pareto front size: {len(final_fronts[0])}")
            return final_fronts[0]
        else:
            print("\n Warning: No Pareto front found!")
        return []   #return an empty list instead of None

    
    def initialize_population(self):
        population = []

        for _ in range(self.pop_size):
            unvisited = set(self.customers)
            routes, current, load = [], [0], 0

            while unvisited:
                # Any customers that still fits capacity
                candidates = [
                    c for c in unvisited
                    if load + self.instance.customers[c]['demand'] <= self.instance.capacity
                ]
                
                if not candidates:
                    # Close current route and start new one
                    current.append(0)
                    routes.append(current)
                    current, load = [0], 0
                    continue

                # Randomly select next customer
                next_customer = random.choice(candidates)
                current.append(next_customer)
                load += self.instance.customers[next_customer]['demand']
                unvisited.remove(next_customer)
            
            # Close the last route
            if current[-1] != 0:
                current.append(0)
            routes.append(current)

            population.append(CVRPSolution(routes, self.instance))
        return population
    
    def create_offspring(self, population):
        offspring = []
        
        while len(offspring) < self.pop_size:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            parent1_tour = routes_to_tour(parent1.routes)
            parent2_tour = routes_to_tour(parent2.routes)

            if random.random() < self.crossover_prob:
                child_tour = ox_crossover(parent1_tour, parent2_tour)
            else:
                child_tour = parent1_tour[:] # Clone parent1

            child_tour = mutate_swap(child_tour, self.mutation_prob)
            child_routes = tour_to_routes(child_tour, self.instance)

            child = CVRPSolution(child_routes, self.instance)
            
            offspring.append(child)
        
        return offspring
    
    def tournament_selection(self, population, tournament_size=2):
        """Binary tournament selection"""
        participants = random.sample(population, tournament_size)
        
        # Select based on rank first, then crowding distance
        best = participants[0]
        for p in participants[1:]:
            if p.rank < best.rank:
                best = p
            elif p.rank == best.rank and p.crowding_distance > best.crowding_distance:
                best = p
        
        return best
    
    def crossover(self, parent1, parent2):
        """
        Simple crossover: take some routes from parent1, rest from parent2
        Then repair to ensure all customers visited exactly once
        """
        # For now, just return parent1 (TODO: implement proper crossover)
        return parent1
    
    def mutate(self, solution):
        """
        Mutation: swap two customers in random routes
        """
        new_routes = [route[:] for route in solution.routes]
        
        # Pick a random route with more than 2 nodes (depot + customers + depot)
        valid_routes = [i for i, r in enumerate(new_routes) if len(r) > 3]
        if valid_routes:
            route_idx = random.choice(valid_routes)
            route = new_routes[route_idx]
            
            # Swap two random customers (skip depot at start and end)
            if len(route) > 3:
                i = random.randint(1, len(route) - 3)
                j = random.randint(1, len(route) - 3)
                route[i], route[j] = route[j], route[i]
        
        return CVRPSolution(new_routes, self.instance)
    
    def calculate_crowding_distance(self, front):
        """Calculate crowding distance for solutions in a front"""
        if len(front) <= 2:
            for sol in front:
                sol.crowding_distance = float('inf')
            return
        
        # Initialize
        for sol in front:
            sol.crowding_distance = 0
        
        # For each objective
        objectives = ['total_distance', 'route_balance']
        for obj in objectives:
            # Sort by objective
            front.sort(key=lambda x: getattr(x, obj))
            
            # Boundary solutions get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_min = getattr(front[0], obj)
            obj_max = getattr(front[-1], obj)
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Assign crowding distance
            for i in range(1, len(front) - 1):
                distance = (getattr(front[i+1], obj) - getattr(front[i-1], obj)) / obj_range
                front[i].crowding_distance += distance
    
    def select_next_generation(self, fronts):
        """Select next generation using fronts and crowding distance"""
        new_population = []
        
        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                new_population.extend(front)
            else:
                # Sort by crowding distance and take the best
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining = self.pop_size - len(new_population)
                new_population.extend(front[:remaining])
                break
        
        return new_population