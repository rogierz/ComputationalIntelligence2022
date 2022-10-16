# Lab 1: Set Covering

## Task

Given a number $N$ and some lists of integers $P = (L_0, L_1, L_2, ..., L_n)$,
determine if possible a set of subsets $S = (L_{s_0}, L_{s_1}, L_{s_2}, ..., L_{s_n})$
such that each number between $0$ and $N-1$ appears in at least one list

$$\forall n \in [0, N-1] \ \exists i : n \in L_{s_i}$$

and that the total numbers of elements in all $L_{s_i}$ is minimum. (i.e. $\min \sum_i^{n}len(L_{s_i})$)

## Problem Representation

Starting from the number $N$ we can define a set $U = \{0, 1, ..., N-1\}$ containing all the numbers between 0 and N
