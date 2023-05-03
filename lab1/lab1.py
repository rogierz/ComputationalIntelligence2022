# %% [markdown]
# # Lab 1: Set Covering
#
# First lab + peer review. List this activity in your final report, it will be part of your exam.
#
# ## Task
#
# Given a number $N$ and some lists of integers $P = (L_0, L_1, L_2, ..., L_n)$,
# determine is possible $S = (L_{s_0}, L_{s_1}, L_{s_2}, ..., L_{s_n})$
# such that each number between $0$ and $N-1$ appears in at least one list
#
# $$\forall n \in [0, N-1] \ \exists i : n \in L_{s_i}$$
#
# and that the total numbers of elements in all $L_{s_i}$ is minimum. ($\min \sum_i^{n}len(L_{s_i})$)
#
# ## Instructions
#
# * Create the directory `lab1` inside the course repo (the one you registered with Andrea)
# * Put a `README.md` and your solution (all the files, code and auxiliary data if needed)
# * Use `problem` to generate the problems with different $N$
# * In the `README.md`, report the the total numbers of elements in $L_{s_i}$ for problem with $N \in [5, 10, 20, 100, 500, 1000]$ and the total number on $nodes$ visited during the search. Use `seed=42`.
# * Use `GitHub Issues` to peer review others' lab
#
# ## Notes
#
# * Working in group is not only allowed, but recommended (see: [Ubuntu](https://en.wikipedia.org/wiki/Ubuntu_philosophy) and [Cooperative Learning](https://files.eric.ed.gov/fulltext/EJ1096789.pdf)). Collaborations must be explicitly declared in the `README.md`.
# * [Yanking](https://www.emacswiki.org/emacs/KillingAndYanking) from the internet is allowed, but sources must be explicitly declared in the `README.md`.
#
# **Deadline**
#
# * Sunday, October 16th 23:59:59 for the working solution
# * Sunday, October 23rd 23:59:59 for the peer reviews

# %% [markdown]
# # Code

# %%
import random
import numpy as np
import logging
import pprint
from utils import search, State
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")

# %%
DEBUG = False
logging.getLogger().setLevel(logging.DEBUG if DEBUG else logging.INFO)

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

# %%
INITIAL_STATE = State(())

# %%


def possible_actions(state: State):
    for p in P:
        data = np.zeros(N)
        data[p] = 1

        yield np.array(p)


def result(state: State, action):
    action_bit = np.zeros(N, dtype=bool)
    action_bit[action] = True
    if len(state.data) == 0:
        return State((action_bit, action))
    result_history = np.logical_or(state.data[0], action_bit)
    del action_bit
    return State((result_history, action))


def goal_test(state: State):
    if len(state.data) == 0:
        return False
    return state.data[0].all()


# %%
parent_state = dict()
state_cost = dict()


def h(state):
    return N - state.data[0].sum()


for N in [5, 10, 20]:
    P = problem(N, SEED)
    logging.debug(f"P: {P}")
    U = np.arange(N)
    UxS = np.zeros((N, len(P)), dtype=np.uint8)
    for i, p in enumerate(P):
        UxS[p, i] = 1

    if not UxS.any(axis=1).all():
        logging.info("Impossible")
        exit

    final = search(
        INITIAL_STATE,
        goal_test=goal_test,
        parent_state=parent_state,
        state_cost=state_cost,
        priority_function=lambda s: state_cost[s] + h(s),
        unit_cost=lambda a: len(a),
        possible_actions=possible_actions,
        result=result
    )

    final_prettier = [list(final[i][1]) for i in range(1, len(final))]
    logging.info(
        f"N: {N}, W: {sum(map(lambda x: len(x), final_prettier))}, Solution: {final_prettier}")
    logging.debug(f"Solution: {final_prettier}")
