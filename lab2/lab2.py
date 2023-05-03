# %%
from typing import Optional
import random
import numpy as np
from numpy.typing import NDArray
import logging
from dataclasses import dataclass
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")

# %%
DEBUG = False
logging.getLogger().setLevel(logging.DEBUG if DEBUG else logging.INFO)

# %%


def tournament(population, k=2):
    return max(random.choices(list(population), k=k), key=lambda x: x.fitness)


class Individual:
    P: Optional[NDArray]
    N: Optional[int]

    def __init__(self, *, included_lists: NDArray):
        self.included_lists = included_lists
        self._len = sum(
            map(lambda x: len(x), Individual.P[np.where(self.included_lists == 1)[0]]))
        self.covered_numbers = self._compute_coveredNumbers()
        self.isValid = (self.covered_numbers == 1).all()
        self.isGoal = (self.covered_numbers == 1).all()
        self.bloat = (len(self) - self.N)
        self.fitness = self._compute_fitness()

    def __repr__(self):
        return str(self.included_lists)

    def __len__(self):
        return self._len

    def __hash__(self):
        return hash(bytes(self.included_lists))

    def __matmul__(self, other):
        '''Performs crossover between self and other'''
        assert (Individual.P == other.P).all(
        ), "Two invididuals must belong to the same problem!"
        assert Individual.N == other.N, "Two invididuals must belong to the same problem!"
        index = np.random.randint(len(Individual.P))
        new_included_lists = np.hstack(
            [self.included_lists[:index], other.included_lists[index:]])
        return Individual(included_lists=new_included_lists)

    def __invert__(self):
        new_included_lists = self.included_lists[:]
        indexes = np.random.choice(
            [True, False], p=[0.2, 0.8], size=len(new_included_lists))
        new_included_lists[indexes] = 1 - new_included_lists[indexes]
        return Individual(included_lists=new_included_lists)

    def _list_to_binary(self, list: NDArray):
        binary_mapping = np.zeros(Individual.N, dtype=bool)
        binary_mapping[list] = True
        return binary_mapping

    def _compute_coveredNumbers(self):
        covered_numbers = np.zeros(Individual.N, dtype=bool)
        for i in np.where(self.included_lists == 1)[0]:
            binary_mapping = self._list_to_binary(Individual.P[i])
            covered_numbers = np.logical_or(
                covered_numbers, binary_mapping).astype(bool)
        return covered_numbers

    def _compute_fitness(self):
        if sum(self.covered_numbers) != Individual.N:
            return np.NINF
        individual_metrics = np.array([sum(self.covered_numbers), self.bloat])
        individual_objective = np.array([Individual.N, 0])
        fitness = - \
            np.sqrt(np.sum((individual_metrics - individual_objective)**2))
        return fitness

    @classmethod
    def init(cls, P, N):
        cls.P = P
        cls.N = N

# %%


def problem(N, seed=None):
    random.seed(seed)
    return [
        list(set(random.randint(0, N - 1)
             for n in range(random.randint(N // 5, N // 2))))
        for n in range(random.randint(N, N * 5))
    ]


# %%
SEED = 42
MAX_COUNT_GENERATION = 100
MAX_PATIENCE = 50

# %%
results = {}
for N in [100, 500, 1000]:
    P = np.array(problem(N, seed=42), dtype=object)
    np.random.seed(None)
    logging.info(
        f"Starting with N: {N}, len(P): {sum(map(lambda x: len(x), P))}")
    Individual.init(P, N)
    POPULATION_SIZE = 200
    OFFSPRING_SIZE = POPULATION_SIZE//4
    population = set()

    old_fittest = None
    fittest = None
    initial_population = np.eye(len(P), dtype=bool)
    # for genome in (np.fromiter((np.random.choice([1, 0]) for _ in P), dtype=np.int8) for  in range(POPULATION_SIZE)):
    for i in range(len(P)):
        genome = initial_population[i]
        individual = Individual(included_lists=genome)
        if individual not in population:
            # adding individual in population, the dict is used to leverage the fast in (not in) operator and to have just one copy of the same individual
            population.add(individual)
            # keeping track of fittest individual discovered so far
            if fittest is None or individual.fitness > fittest.fitness:
                fittest = individual
                # initialziation of old_fittest
                old_fittest = fittest

    logging.info(
        f"Starting fittest, fitness: {fittest.fitness:.2f}, W: {len(fittest)}, goal: {fittest.isGoal}")
    count_generation = 0
    patience = 0
    while count_generation < MAX_COUNT_GENERATION and patience < MAX_PATIENCE:
        count_generation += 1
        logging.debug(
            f"----------- [Generation: {count_generation}/{MAX_COUNT_GENERATION}] -----------")

        offspring = set()
        for _ in range(OFFSPRING_SIZE):
            p1, p2 = tournament(population=population, k=5), tournament(
                population=population, k=5)
            o = p1 @ p2  # crossover
            # always add the generated offspring
            offspring.add(o)

            # but let some space to randomness
            if np.random.rand() < .2:
                o = ~o  # mutation
                offspring.add(o)

        # add offspring into population
        population |= offspring
        # poplation as a sorted list to keep the population size constant
        population = sorted(list(population), key=lambda x: x.fitness, reverse=True)[
            :POPULATION_SIZE]

        old_fittest = fittest
        if population[0].fitness > fittest.fitness:
            fittest = population[0]
            logging.debug(
                f"New fittest found. Fitness: {fittest.fitness:.2f}, W: {len(fittest)}, goal: {fittest.isGoal}")
            patience = 0

        if old_fittest == fittest:
            logging.debug(
                f"No improvement, patience {patience}/{MAX_PATIENCE}")
            patience += 1

        if DEBUG is False and count_generation % 10 == 0:
            logging.info(
                f"[{count_generation}/{MAX_COUNT_GENERATION}] Fitness: {fittest.fitness:.2f}, W: {len(fittest)}, goal: {fittest.isGoal}")

        population = set(population)

    logging.info(
        f"N={N}, W={len(fittest)}, bloat={fittest.bloat/N*100:.2f}%, {fittest.isGoal}")
    results[N] = fittest

# %%
for (N, individual) in results.items():
    logging.info(f"N: {N}, {len(individual)}, {individual.isGoal}")
