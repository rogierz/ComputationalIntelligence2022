# %% [markdown]
# # Lab 3: Policy Search
#
# ## Task
#
# Write agents able to play [*Nim*](https://en.wikipedia.org/wiki/Nim), with an arbitrary number of rows and an upper bound $k$ on the number of objects that can be removed in a turn (a.k.a., *subtraction game*).
#
# The player **taking the last object wins**.
#
# * Task3.1: An agent using fixed rules based on *nim-sum* (i.e., an *expert system*)
# * Task3.2: An agent using evolved rules
# * Task3.3: An agent using minmax
# * Task3.4: An agent using reinforcement learning
#
# ## Deadlines ([AoE](https://en.wikipedia.org/wiki/Anywhere_on_Earth))
#
# * Sunday, December 4th for Task3.1 and Task3.2
# * Sunday, December 11th for Task3.3 and Task3.4
# * Sunday, December 18th for all reviews

# %%
from collections import defaultdict
import math
from functools import cache
import logging
import random
import numpy as np
from collections import namedtuple
from typing import Callable
from copy import deepcopy
from functools import reduce
from operator import xor

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.DEBUG)

# %% [markdown]
# ## The *Nim* and *Nimply* classes

# %%
Nimply = namedtuple("Nimply", "row, num_objects")


class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k
        self.player = 0

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    @property
    def k(self) -> int:
        return self._k

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects
        self.player = 1 - self.player

    def __hash__(self):
        return hash(bytes([*self.rows, self.player]))

    def __eq__(self, other):
        return all(map(lambda x: x[0] == x[1], zip(self.rows, other.rows))) and self.player == other.player


def nim_sum(state: Nim) -> int:
    result = reduce(xor, state.rows)
    return result


def cook_status(state: Nim) -> dict:
    cooked = dict()
    cooked["possible_moves"] = [
        (r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1) if state.k is None or o <= state.k
    ]
    cooked["nim_sum"] = nim_sum(state)

    brute_force = list()
    for m in cooked["possible_moves"]:
        tmp = deepcopy(state)
        tmp.nimming(m)
        brute_force.append((m, nim_sum(tmp)))
    cooked["brute_force"] = brute_force

    return cooked


def optimal_strategy(state: Nim) -> Nimply:
    data = cook_status(state)
    return next((bf for bf in data["brute_force"] if bf[1] == 0), data["brute_force"][0])[0]


# %% [markdown]
# ## Task 3.1

# %% [markdown]
# ### My fixed-rule agent

# %%
Genome = namedtuple("Genome", "aggressivity, longest_first, how_many")
Individual = namedtuple("Individual", "genome, fitness")

# %%


def make_strategy(genome: Genome) -> Callable:
    def evolvable(state: Nim) -> Nimply:
        # data = cook_status(state)
        aggressive: bool = random.random() < genome.aggressivity
        longest_first: bool = random.random() < genome.longest_first
        how_many_coeff: float = genome.how_many

        # Sort the rows based on the number of elements, the sort is descending or ascending based on if longest_first or not
        row_indexes = sorted((i for i in range(len(
            state.rows)) if state.rows[i] > 0), key=lambda elem: state.rows[elem], reverse=longest_first)

        # Select randomly one of the first 50% rows
        selected_row_index = random.choice(
            row_indexes[:int(0.5*len(row_indexes))+1])

        # Decide to take at least 1 or half of the objects
        take_at_least = 0 if not aggressive else state.rows[selected_row_index]//2

        # Decide to take or not a part of the remaining objects
        take_n = max(1, min(
            take_at_least + int(state.rows[selected_row_index]//2*how_many_coeff), state.rows[selected_row_index]))
        ply = Nimply(selected_row_index, take_n)

        return ply

    return evolvable


# Task 3.1 agent
not_evovled = make_strategy(Genome(0.5, 0.5, 0.5))

# %% [markdown]
# ### Other fixed-rule agents

# %%


def pure_random(state: Nim) -> Nimply:
    row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
    num_objects = random.randint(1, state.rows[row])
    return Nimply(row, num_objects)


def gabriele(state: Nim) -> Nimply:
    """Pick always the maximum possible number of the lowest row"""
    possible_moves = [(r, o) for r, c in enumerate(state.rows)
                      for o in range(1, c + 1)]
    return Nimply(*max(possible_moves, key=lambda m: (-m[0], m[1])))

# %% [markdown]
# ## Task 3.2

# %% [markdown]
# The agent base policy it's the same of task 3.1, but now I evolve its parameters


# %%
NIM_SIZE = 10


def evaluate(strategy: Callable, opponent: Callable = optimal_strategy, NUM_MATCHES=100) -> float:
    strategies = (strategy, opponent)
    won = 0

    for m in range(NUM_MATCHES):
        nim = Nim(NIM_SIZE)
        player = 0
        while nim:
            ply = strategies[player](nim)
            nim.nimming(ply)
            player = 1 - player
        if player == 1:
            won += 1
    return won / NUM_MATCHES

# %% [markdown]
# ### Evolve genome

# %%


def tournament(population, k=10):
    return max(random.choices(population, k=k), key=lambda x: x.fitness)


def make_offspring(p1: Individual, p2: Individual, current_best_strategy):
    new_genome = []
    for i in range(len(p1.genome)):
        # inherit from p1 or p2 based on randomness
        gene = p1.genome[i] if random.random() > 0.5 else p2.genome[i]
        gene += random.gauss(0, 0.25)  # tweak
        gene = max(0, min(gene, 1))  # clip to avoid unammissible solutions
        new_genome.append(gene)

    new_genome = Genome(*new_genome)
    individual = Individual(new_genome, evaluate(make_strategy(
        new_genome), current_best_strategy))  # create individual and compute fitness

    return individual


# %%
ITERATIONS = 20
POPULATION_SIZE = 40
OFFSPRING_SIZE = 200
genomes = [Genome(0.5 + random.random()/10, 0.5 + random.random()/10,
                  0.5 + random.random()/10) for _ in range(POPULATION_SIZE)]

# the initial population fitness is computed against gabriele
population = list(map(lambda genome: Individual(
    genome, evaluate(make_strategy(genome), gabriele)), genomes))

for i in range(ITERATIONS):
    logging.debug(
        f"Starting iteration {i}. Current fitness: {population[0].fitness}")
    logging.debug(f"Current best:\nAggressivity\tLongest first\tHow many\n\
{population[0].genome.aggressivity:.2f}\t\t{population[0].genome.longest_first:.2f}\t\t{population[0].genome.how_many:.2f}")

    offspring = []
    for _ in range(OFFSPRING_SIZE):
        p1, p2 = tournament(population, k=1), tournament(population, k=1)
        o = make_offspring(p1, p2, make_strategy(population[0].genome))
        offspring.append(o)
    population.extend(offspring)
    population = sorted(population, key=lambda individual: individual.fitness, reverse=True)[
        :POPULATION_SIZE]

evolved_individual = population[0]

# %% [markdown]
# ### Task 3.2 final match

# %%
strategies = (make_strategy(evolved_individual.genome), optimal_strategy)

nim = Nim(11)
logging.debug(f"status: Initial board  -> {nim}")
player = 0
while nim:
    ply = strategies[player](nim)
    nim.nimming(ply)
    logging.debug(f"status: After player {player} -> {nim}")
    player = 1 - player
winner = 1 - player
logging.info(f"status: Player {winner} won!")

# %% [markdown]
# ## Task 3.3

# %%


@cache
def possible_actions(state):
    possible_moves = [(r, o) for r, c in enumerate(state.rows)
                      for o in range(1, c + 1)]
    return possible_moves


@cache
def result(state: Nim, action: Nimply):
    new_state = deepcopy(state)
    new_state.nimming(action)
    return new_state

# https://mathspp.com/blog/minimax-algorithm-and-alpha-beta-pruning


def minmax_strategy_ab(state: Nim):

    def minmax(state: Nim, alpha=-1, beta=1):
        maximising_player = state.player == 0

        if not state:
            # no more moves: state.player lost
            return None, 1 if not maximising_player else -1

        val = (None, -1) if maximising_player else (None, 1)
        for ply in cook_status(state)["possible_moves"]:
            new_state = result(state, ply)
            _, ns_value = minmax(new_state, alpha, beta)

            if maximising_player:
                val = max((ply, ns_value), val, key=lambda x: x[1])
                alpha = max(alpha, ns_value)
            else:
                val = min((ply, ns_value), val, key=lambda x: x[1])
                beta = min(beta, ns_value)

            if (maximising_player and val[1] >= beta) or (not maximising_player and val[1] <= alpha):
                break
        return val
    return minmax(state)[0]

# %% [markdown]
# ## Task 3.4


# %%


class Agent():
    def __init__(self, alpha=0.15, explore_factor=0.8):  # 80% explore, 20% exploit
        self.state_history = []  # state, reward
        self.alpha = alpha
        self.explore_factor = explore_factor
        self.G = defaultdict(lambda: np.random.uniform(low=0.1, high=1))

    def choose_action(self, state, allowedMoves, train=True):
        maxG = -10e15
        next_move = None
        randomN = np.random.random()

        explore_factor = self.explore_factor if train else 0.05

        if randomN < explore_factor:
            # if random number below random factor, choose random action
            # i need to do this since allowedMoves is a list of tuples and it messes up numpy
            next_move = allowedMoves[np.random.choice(len(allowedMoves))]
        else:
            # if exploiting, gather all possible actions and choose one with the highest G (reward)
            for action in allowedMoves:
                new_state = deepcopy(state)
                new_state.nimming(action)
                if self.G[new_state] >= maxG:
                    next_move = action
                    maxG = self.G[new_state]

        return next_move

    def update_state_history(self, state, reward):
        self.state_history.append((state, reward))

    def learn(self):
        target = 0

        for prev, reward in reversed(self.state_history):
            self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])
            target += reward

        self.state_history = []

        self.explore_factor -= 10e-5  # decrease random factor each episode of play

    def __call__(self, state):
        allowed_moves = cook_status(state)["possible_moves"]
        return self.choose_action(state, allowed_moves)


# %%
# Train agent against optimal
GAME_LENGTH = 4
def fact(n): return n*(n-1)


robot = Agent()
for i in range(100000):
    player = 0
    nim = Nim(GAME_LENGTH)
    while nim:
        if player == 1:
            ply = optimal_strategy(nim)
            nim.nimming(ply)
            player = 1 - player
            continue

        action = robot(nim)
        nim.nimming(action)
        state, reward = nim, -1 if nim else fact(GAME_LENGTH)
        robot.update_state_history(state, reward)
        player = 1 - player
    robot.learn()

# %%
strategies = (robot, optimal_strategy)

nim = Nim(4)
logging.debug(f"status: Initial board  -> {nim}")
player = 0
while nim:
    ply = strategies[player](nim)
    nim.nimming(ply)
    logging.debug(f"status: After player {player} -> {nim}")
    player = 1 - player
winner = 1 - player
logging.info(f"status: Player {winner} won!")
