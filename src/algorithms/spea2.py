import random
import numpy as np
from src.algorithms.utils import fast_non_dominated_sort
from src.operators.ox import ox_crossover, mutate_swap
from src.problem.split import routes_to_tour, tour_to_routes
from src.problem.solution import CVRPSolution
from operator import attrgetter

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
    
    def _assign_fitness(self, union):
        n = len(union)

        # Strength S(i): number of solutions it dominates
        
        S=np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and dominates(union[i], union[j]):
                    S[i] += 1

        # Raw fitness R(i): sum of strengths of solutions that dominate it'
        R = np.zeros(n)
        for i in range(n):
            total = 0.0
            for j in range(n):
                if i != j and dominates(union[j], union[i]):
                    total += S[j]
            R[i] = total

        
        #Density D(i): normalized space so distance and balanbce are comparable
        k = max(1, int(np.sqrt(n)))
        f1s = [s.total_distance for s in union]
        f2s = [s.route_balance for s in union]
        f1min, f1max = min(f1s), max(f1s)
        f2min, f2max = min(f2s), max(f2s)

        normalized = ([
            [
                (s.total_distance - f1min) / (f1max - f1min) if f1max > f1min else 0,
                (s.route_balance - f2min) / (f2max - f2min) if f2max > f2min else 0
            ]
            for s in union
        ])

        D = np.zeros(n)
        for i in range(n):
            xi, yi = normalized[i]
            distances = []
            for j in range(n):
                if i == j: continue
                xj, yj = normalized[j]
                dx, dy = xi - xj, yi - yj
                distances.append(np.sqrt(dx * dx + dy * dy))
            if distances:
                distances.sort()
                # Use k-th neighbor if available, else the furtheest
                sigma_k = distances[k - 1] if len(distances) >= k else distances[-1]
            else:
                sigma_k = 0.0

            D[i] = 1 / (sigma_k + 2)  # Avoid division

        # Overall fitness F(i) = R(i) + D(i)

        F = [R[i] + D[i] for i in range(n)]
        
        for i, solution in enumerate(union):
            solution.fitness = F[i]

        return {union[i]: F[i] for i in range(n)}
    
    def _environmental_selection(self, union, fitness, size):
        # Select all non-dominated solutions (Fitness < 1)
        selected = {solution for solution in union if fitness[solution] < 1.0}

        # Fill with best fitness solutions if needed
        if len(selected) < size:
            rest = sorted(solution for solution in union if solution not in selected)
            rest.sort(key=attrgetter('fitness', 'total_distance', 'route_balance'))
            need = size - len(selected)
            selected.extend(rest[:need])

        # If too many, remove the most crowded points
        while len(selected) > size:
            f1s = [s.total_distance for s in selected]
            f2s = [s.route_balance for s in selected]
            f1min, f1max = min(f1s), max(f1s)
            f2min, f2max = min(f2s), max(f2s)
            
            # Normalize

            normalized = [(
                (s.total_distance - f1min) / (f1max - f1min) if f1max > f1min else 0,
                (s.route_balance - f2min) / (f2max - f2min) if f2max > f2min else 0
            ) for s in selected]

            # Remove the one with smallest distance to any other
            rm_idx, best_dist = 0, float('inf')
            m = len(selected)
            for i in range(m):
                xi, yi = normalized[i]
                min_dist = float('inf')
                for j in range(m):
                    if i == j: continue
                    xj, yj = normalized[j]
                    d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    if d < min_dist:
                        min_dist = d
                if min_dist < best_dist:
                    best_dist = min_dist
                    rm_idx = i
            selected.remove(list(selected)[rm_idx])
            
        return list(selected)