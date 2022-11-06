# Lab 2

## Set covering with GA

The task is the same as lab #1. Here the solution propoesd is based on a genetich algorithm.

The individual is encoded as a binary array, each element indicates if the i-th list is included in the set or not.

At each mutation step, each locus has the possiblity to flip, in contrast with the original implementation in which only one locus could.

The crossover is performed between two parents, each one is selected with a tournament as in the onemax problem discussed with the professor.

The offspring generated, is immediately added to the population, and after the offspring generation only the best `POPULATION_SIZE` individuals survive.

Unfortunately, this implementation doesn't performs so well, so further improvements are needed.
