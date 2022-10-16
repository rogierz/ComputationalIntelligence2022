# Lab 1: Set Covering

## Task

Given a number $N$ and some lists of integers $P = (L_0, L_1, L_2, ..., L_n)$,
determine if possible a set of subsets $S = (L_{s_0}, L_{s_1}, L_{s_2}, ..., L_{s_n})$
such that each number between $0$ and $N-1$ appears in at least one list

$$\forall n \in [0, N-1] \ \exists i : n \in L_{s_i}$$

and that the total numbers of elements in all $L_{s_i}$ is minimum. (i.e. $\min \sum_i^{n}len(L_{s_i})$)

## State Encoding

Given $N$ we define:

The initial state as an empty tuple.

Generic state as the tuple:

$S = (s_0, s_1)$

Where

- $s_0 = $ binary map of missing numbers in the state (e.g. `[1, 0, 0, 1]` stands for `[[0, 3]]` as well as `[[0], [3]]`)

- $s_1 = $ last subset (action) added to generate the new state

## Search strategy

The `search` function implemented follows a standard strategy:

1. Start from a node
2. Check if the current node is the goal
3. Generate the nodes reachable from the current node
4. Choose the next node to process
5. Repeat from (2)

### Building the frontier

To generate the candidates we use the generator `possible_action(state: State)`. This yields one list among the ones contained in $P$ (defined in the [Task](#task) section).

To check if the list has already been included it performs a `logical and` between the binary mapping of the list (e.g. `[1, 3]` become `[0, 1, 0, 1]`) and the $s_0$ element of the current state. If the result of this operation is the binary mapping itself, then it's not useful to consider this list.

Given each candidate we build the new state to add to the frontier as:

$S_{n} = (OR(S_{n-1, 0},a_n),\ a_{n})$

### Choosing the next node to process

We sort the nodes on the frontier according to a priority function that includes:

- the cost of the actual state (that is equal to the sum of the lenghts of the actions nedded to reach the state)
- the cost given by an heuristic function that computes the Hamming distance between the current state and the goal state

(Here I'm trying to implement the A^\*^ algorithm, but I'm not sure that the heuristic function is _admissible_)

## Experimental results:

|  N   | $\sum$ | Node visited |
| :--: | :----: | :----------: |
|  5   |   5    |      3       |
|  10  |   10   |      3       |
|  20  |   20   |     4189     |
| 100  |   ~    |      ~       |
| 500  |   ~    |      ~       |
| 1000 |   ~    |      ~       |
