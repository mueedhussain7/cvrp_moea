"""
Utility functions for MOEAs
"""
def dominates(sol1, sol2):
    """
    Check if sol1 dominates sol2 (Pareto dominance for minimization)
    
    sol1 dominates sol2 if:
    - sol1 is no worse than sol2 in all objectives
    - sol1 is strictly better than sol2 in at least one objective
    """
    better_in_at_least_one = False
    
    # Check both objectives
    if sol1.total_distance < sol2.total_distance:
        better_in_at_least_one = True
    elif sol1.total_distance > sol2.total_distance:
        return False
    
    if sol1.route_balance < sol2.route_balance:
        better_in_at_least_one = True
    elif sol1.route_balance > sol2.route_balance:
        return False
    
    return better_in_at_least_one


def fast_non_dominated_sort(population):
    """
    NSGA-II fast non-dominated sorting
    
    Returns:
        List of fronts, where each front is a list of solutions
    """
    # Initialize
    for p in population:
        p.dominated_by = []
        p.dominates_count = 0
    
    # Find domination relationships
    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i != j:
                if dominates(p, q):
                    p.dominated_by.append(q)
                elif dominates(q, p):
                    p.dominates_count += 1
    
    # Build fronts
    fronts = [[]]
    for p in population:
        if p.dominates_count == 0:
            p.rank = 1
            fronts[0].append(p)
    
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_by:
                q.dominates_count -= 1
                if q.dominates_count == 0:
                    q.rank = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    return fronts[:-1]  # Remove empty last front


if __name__ == '__main__':
    # Test dominance
    from src.problem.cvrp_reader import read_cvrp_file
    from src.problem.solution import CVRPSolution
    
    instance = read_cvrp_file('../../data/A-n32-k5.vrp')
    
    sol1 = CVRPSolution([[0, 2, 3, 0], [0, 4, 5, 0]], instance)
    sol2 = CVRPSolution([[0, 2, 3, 4, 0], [0, 5, 0]], instance)
    
    print(f"Solution 1: {sol1}")
    print(f"Solution 2: {sol2}")
    print(f"Sol1 dominates Sol2: {dominates(sol1, sol2)}")