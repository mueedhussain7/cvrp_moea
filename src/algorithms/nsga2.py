"""
NSGA-II implementation for CVRP
"""
import random
import numpy as np
from src.algorithms.utils import fast_non_dominated_sort
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
                 crossover_prob=0.7, mutation_prob=0.2):
        self.instance = instance
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Get list of customer IDs
        self.customers = list(instance.customers.keys())
        
    def run(self):
        print(f"Initializing population of {self.pop_size}...")
        population = self.initialize_population()

        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            self.calculate_crowding_distance(front)

        for gen in range(self.generations):
            offspring = self.create_offspring(population)
            combined = population + offspring

            fronts = fast_non_dominated_sort(combined)
        for front in fronts:
            self.calculate_crowding_distance(front)

        population = self.select_next_generation(fronts)

        if (gen + 1) % 50 == 0:
            print(f"Generation {gen + 1}/{self.generations} - Front 1 size: {len(fronts[0])}")

            final_fronts = fast_non_dominated_sort(population)
        if final_fronts and len(final_fronts[0]) > 0:
            print(f"\nFinal Pareto front size: {len(final_fronts[0])}")
            return final_fronts[0]
        else:
            print("\n Warning: No Pareto front found!")
        return []   #return an empty list instead of None

    
    def initialize_population(self):
        """Create initial random population"""
        population = []
        for _ in range(self.pop_size):
            solution = self.create_random_solution()
            population.append(solution)
        return population
    
    def create_random_solution(self):
        """Create a random feasible solution using nearest neighbor heuristic"""
        unvisited = set(self.customers)
        routes = []
        
        while unvisited:
            route = [0]  # Start at depot
            current = 0
            current_load = 0
            
            while unvisited:
                # Find nearest unvisited customer that fits capacity
                candidates = []
                for customer in unvisited:
                    demand = self.instance.customers[customer]['demand']
                    if current_load + demand <= self.instance.capacity:
                        candidates.append(customer)
                
                if not candidates:
                    break
                
                # Choose randomly from candidates (for diversity)
                next_customer = random.choice(candidates)
                
                route.append(next_customer)
                current_load += self.instance.customers[next_customer]['demand']
                unvisited.remove(next_customer)
                current = next_customer
            
            route.append(0)  # Return to depot
            routes.append(route)
        
        return CVRPSolution(routes, self.instance)
    
    def create_offspring(self, population):
        """Create offspring through selection, crossover, and mutation"""
        offspring = []
        
        while len(offspring) < self.pop_size:
            # Tournament selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            if random.random() < self.crossover_prob:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1  # Keep parent
            
            # Mutation
            if random.random() < self.mutation_prob:
                child = self.mutate(child)
            
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