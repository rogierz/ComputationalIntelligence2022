# Lab 2

## Nim

## Task 3.1

The agent follows a policy with pre-defined parameters, that are: aggressivity, longest_first, how_many.
Each of the parameter characterize the playstyle of the agent. More in detail:

- `aggressivity` is the probability of the agent to take more of the 50% of the elements of a row
- `longest_first` is the probability of the agent to select longer rows, the choice's criterion is sorting the row based on the length ascending or descending according to `longest_first`, then select one among the first half of the sorted rows
- `how_many` is the percentage of elements to take of the selected half

## Task 3.2

Standard population-based evolutionary algorithm, the fitness is computed using a tournament among the individuals. Each individual plays a number of
matches with the current best, the percentage of matches won by the generated individual is its fitness.

**N.B.**: the fitness is computed against the current best and not against the nim-sum due to the difficulty to evolve the population.

## Task 3.3

Minmax implementation with alpha-beta pruning

## Task 3.4

RL-based implementation
