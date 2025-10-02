import random

def ox_crossover(parent1, parent2):
    n = len(parent1)

    # Choose two random crossover points
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n

    # Copy segment from parent1 to child
    child[a:b+1] = parent1[a:b+1]

    # Fill remaining positions with parent2 genes, skipping dublicates

    fill = [gene for gene in parent2 if gene not in child]
    fill_index = 0

    for i in range(n):
        if child[i] is None:
            child[i] = fill[fill_index]
            fill_index += 1
    return child

def mutate_swap(tour, p=0.2):
    
    if len(tour) >= 2 and random.random() < p:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour