import random
import numpy as np
from src.algorithms.utils import fast_non_dominated_sort
from src.operators.ox import ox_crossover, mutate_swap
from src.problem.split import routes_to_tour, tour_to_routes
from src.problem.solution import CVRPSolution

def dominates(a, b):
    # Solution a dominates b if a is no worse in both and at least better in one objective (minimzation of distance and fairness)
    
    return (
        (a.total_distance <= b.total_distance) and
        (a.route_balance <= b.route_balance) and
        ((a.total_distance < b.total_distance) or
         (a.route_balance < b.route_balance)))


class SPEA2:
    
    def __init__(self, instance, pop_size=100, generations=500, 
                 crossover_prob=0.7, mutation_prob=0.2, seed=None):
                 
        self.instance = instance
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.seed = seed

        self.customers = list(instance.customers.keys())

    def run(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        print(f"Intializting population of {self.pop_size}...")
        population = self.initialize_population()
        archive = []

        for gen in range(self.generations):
            combined = population + archive
            fitness = self._assign_fitness(combined)
            archive = self._environmental_selection(combined, fitness, self.pop_size)
            population = self._mating_and_variation(archive)

            if (gen + 1) % 10 == 0 or gen == self.generations - 1:
                fronts = fast_non_dominated_sort(archive)
                f1 = len(fronts[0]) if fronts else 0
                print(f"Generation {gen + 1}/{self.generations} - Archive size: {len(archive)} - F1: {f1}")

        final_fronts = fast_non_dominated_sort(archive)
        if final_fronts and len(final_fronts[0]) > 0:
            print(f"\nFinal Pareto front size: {len(final_fronts[0])}")
            return final_fronts[0]
        print("\nWarning (SPEA2): No pareto front found.")
        return []
    
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
    
    def _assign_fitness(self, combined):