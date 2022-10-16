import logging
import heapq
import numpy as np
from functools import reduce
from typing import Callable


class PriorityQueue:
    """A basic Priority Queue with simple performance optimizations"""

    def __init__(self):
        self._data_heap = list()
        self._data_set = set()

    def __bool__(self):
        return bool(self._data_set)

    def __contains__(self, item):
        return item in self._data_set

    def push(self, item, p=None):
        assert item not in self, f"Duplicated element"
        if p is None:
            p = len(self._data_set)
        self._data_set.add(item)
        heapq.heappush(self._data_heap, (p, item))

    def pop(self):
        p, item = heapq.heappop(self._data_heap)
        self._data_set.remove(item)
        return item


class State:
    def __init__(self, data: tuple):
        self._data = data

    def __hash__(self):
        return hash(tuple(bytes(t) for t in self._data))

    def __eq__(self, other):
        first_inner = self._data[0] == other._data[0]
        second_inner = self._data[1] == other._data[1]

        first = first_inner if type(first_inner) is bool else first_inner.all()
        second = second_inner if type(
            second_inner) is bool else second_inner.all()

        return first and second

    def __lt__(self, other):
        return self._data[0].sum() > other._data[0].sum()

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return repr(self._data)

    @property
    def data(self):
        return self._data


def search(
    initial_state: State,
    goal_test: Callable,
    parent_state: dict,
    state_cost: dict,
    priority_function: Callable,
    unit_cost: Callable,
    possible_actions: Callable,
    result: Callable
):
    frontier = PriorityQueue()
    parent_state.clear()
    state_cost.clear()

    state = initial_state
    parent_state[state] = None
    state_cost[state] = 0
    visited_counter = 0
    while state is not None and not goal_test(state):
        logging.debug(f"Entering in state: {state.data}")
        logging.debug(f"Building frontier...")
        # Build the frontier
        for a in possible_actions(state):

            new_state = result(state, a)
            cost = unit_cost(a)

            if new_state not in state_cost and new_state not in frontier:
                parent_state[new_state] = state
                state_cost[new_state] = state_cost[state] + cost
                frontier.push(new_state, p=priority_function(new_state))
                logging.debug(
                    f"Added new node to frontier (cost={state_cost[new_state]})")
            elif new_state in frontier and state_cost[new_state] > state_cost[state] + cost:
                old_cost = state_cost[new_state]
                parent_state[new_state] = state
                state_cost[new_state] = state_cost[state] + cost
                logging.debug(
                    f"Updated node cost in frontier: {old_cost} -> {state_cost[new_state]}")

        # Get the next node to visit
        if frontier:
            state = frontier.pop()
            visited_counter += 1
        else:
            state = None

    path = list()
    s = state
    while s:
        path.append(s.data)
        s = parent_state[s]

    logging.info(
        f"Found a solution in {len(path):,} steps; visited {visited_counter} nodes")
    return list(reversed(path))
