"""
Configuration for experiments
"""

# Parameter sets for testing
PARAMETER_SETS = [
    {
        'name': 'default',
        'pop_size': 100,
        'generations': 500,
        'crossover_prob': 0.7,
        'mutation_prob': 0.2
    },
    {
        'name': 'high_mutation',
        'pop_size': 100,
        'generations': 500,
        'crossover_prob': 0.7,
        'mutation_prob': 0.4
    },
    {
        'name': 'large_population',
        'pop_size': 200,
        'generations': 250,
        'crossover_prob': 0.7,
        'mutation_prob': 0.2
    }
]

# Instances to test
INSTANCES = {
    'small': 'data/A-n32-k5.vrp',
}