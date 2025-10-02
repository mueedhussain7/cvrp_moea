"""
Main entry point for running MOVRP experiments
"""
import argparse
import time
from src.problem.cvrp_reader import read_cvrp_file
from src.algorithms.nsga2 import NSGA2
from src.algorithms.spea2 import SPEA2
from src.problem.solution import CVRPSolution


def main():
    parser = argparse.ArgumentParser(description='Multi-Objective CVRP Solver')
    parser.add_argument('--algorithm', choices=['nsga2', 'spea2'], 
                       default='nsga2', help='Algorithm to use')
    parser.add_argument('--instance', required=True, 
                       help='Path to CVRP instance file')
    parser.add_argument('--runs', type=int, default=1, 
                       help='Number of independent runs')
    parser.add_argument('--pop-size', type=int, default=100, 
                       help='Population size')
    parser.add_argument('--generations', type=int, default=500, 
                       help='Number of generations')
    parser.add_argument('--crossover-prob', type=float, default=0.7, 
                       help='Crossover probability')
    parser.add_argument('--mutation-prob', type=float, default=0.2, 
                       help='Mutation probability')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (for reproducibility)')
    
    args = parser.parse_args()
    
    # Load problem instance
    print(f"Loading instance: {args.instance}")
    instance = read_cvrp_file(args.instance)
    print(instance)
    print(f"Capacity: {instance.capacity}")
    print(f"Customers: {len(instance.customers)}\n")
    
    # Run experiments
    all_fronts = []
    for run in range(args.runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{args.runs}")
        print(f"{'='*60}")
        
        CVRPSolution.EVAL_COUNTER = 0 # Reset evaluation counter each run
        start_time = time.time()
        
        if args.algorithm == 'nsga2':
            algorithm = NSGA2(
                instance,
                pop_size=args.pop_size,
                generations=args.generations,
                crossover_prob=args.crossover_prob,
                mutation_prob=args.mutation_prob,
                seed=args.seed
            )
        elif args.algorithm.lower() == 'spea2':
            algorithm = SPEA2(
                instance,
                pop_size=args.pop_size,
                generations=args.generations,
                crossover_prob=args.crossover_prob,
                mutation_prob=args.mutation_prob,
                seed=getattr(args, "seed", None)
            )

        
        pareto_front = algorithm.run()
        
        elapsed = time.time() - start_time
        
        all_fronts.append(pareto_front)
        evals = CVRPSolution.EVAL_COUNTER
        
        print(f"\nRun completed in {elapsed:.2f} seconds")
        print(f"Pareto front size: {len(pareto_front)}")
        
        # Display some solutions
        print("\nSample solutions from Pareto front:")
        for i, sol in enumerate(pareto_front[:5]):
            print(f"  {i+1}. Distance: {sol.total_distance:.2f}, "
                  f"Balance: {sol.route_balance:.2f}, "
                  f"Routes: {len(sol.routes)}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Instance: {instance.name}")
    print(f"Runs completed: {len(all_fronts)}")
    print(f"Average Pareto front size: {sum(len(f) for f in all_fronts) / len(all_fronts):.1f}")
    print(f"Total evaluations across all runs: {sum(CVRPSolution.EVAL_COUNTER for _ in all_fronts)}")

if __name__ == '__main__':
    main()