import random as rand
import numpy as np
import sys
import math
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

MAXITER = 0  # holds the maximal number of iterations in our algorithm. sanity check
MAX_BIDDERS = 3  # maximal number of bidders. The actual number of bidders is randomized out of domain (1,MAX_BIDDERS)
MAX_ITEMS = (MAX_BIDDERS * 2)  # maximal number of items. The actual number of items is randomized later in code
MAX_VALUATION = 1000  # maximal possible valuation a bidder can have for some item. again the actual number is randomized.
NUM_TESTS = 500  # number of tests to run.

INVALID_VALUATION = MAX_VALUATION * MAX_ITEMS + 1

# ------------------------------------- UNUSED - MULTITHREADED
# import os
# import multiprocessing as mp
# from multiprocessing import freeze_support
# from functools import partial

# NUM_THREADS = 8 # this is currently unused, in theory for multithreaded running
# def partition(l, k):
#     n=l//k
#     for i in range(0, l, n):
#         if (i + n > l):
#             yield (i,l)
#         else:
#             yield (i,i + n)

# def calc_max_parallel(n,m,R,L,V):
#     max_range = 2**(m*n)
#     with mp.Pool(NUM_THREADS) as pool:
#         # results = pool.map(calculate_optimal_by_range, (partition(MAX_NUM, NUM_THREADS),n,m,R,L,V))
#         results = pool.map(partial(calculate_optimal_by_range, n=n,m=m,R=R,L=L,V=V), partition(max_range, NUM_THREADS))
#     pool.join()
#     return max(results)
# -------------------------------------


def run_test():
    # Create a bipartite graph with n vertex on the Left side and m vertexes on the right side
    # The graph is represented as a list of lists
    # The first n lists are the vertexes on the left side
    # The second n lists are the vertexes on the right side
    n = rand.randint(1, MAX_BIDDERS)
    m = rand.randint(n, MAX_ITEMS)
    L = [i for i in range(n)]
    R = [i for i in range(m)]
    V = [[MAX_VALUATION * rand.random() for j in range(m)] for i in range(n)]

    optimal_max = calculate_optimal_by_range((0, 2 ** (m * n)), n, m, R, L, V)
    our_algorithm = calculate_our_algorithm(n, m, R, L, V)
    return our_algorithm, optimal_max


# go over all possible allocations for m items to n bidders and calculate the lowest value bidder for each one:
def calculate_optimal_by_range(ranges, n, m, R, L, V):
    max_min_value = 0
    for i in range(ranges[0], ranges[1]):
        # The number i represents the allocation, if bit j is set in i then the item j%m goes to the j/m bidder.
        # Of course this is NOT an optimal way of going over all possible allocations as this goes over invalid allocations where 2 bidders get the same item. We eliminate these cases individually.
        l = i
        allocation = [0 for j in range(m)]
        for j in range(m):
            # this loop takes the number i and turns it into an easy to work with allocation list where in the j-th cell appears the number of the bidder who gets the j-th item
            if l == 0:
                allocation[j] = 0
                continue
            elif len(bin(l)) <= 2 + n:
                allocation[j] = int(bin(l)[2:], 2)
                l = 0
            else:
                allocation[j] = int(bin(l)[-n:], 2)
                l = l >> n
            if allocation[j] > n:
                allocation[j] = 0
        val = INVALID_VALUATION
        for j in range(n):
            # this loop finds the bidder with the minimal total valuation for the allocation
            sum = 0
            for k in range(m):
                if allocation[k] == j:
                    sum += V[j][k]
            val = min(val, sum)
        if val != INVALID_VALUATION:
            max_min_value = max(max_min_value, val)
    return max_min_value


# function to find the bidder with the minimum sum of valuations
def find_min_bidder(n, m, allocation, V):
    val = INVALID_VALUATION
    min_bidder = -1
    sum_of_valuations_by_bidder = [0 for j in range(n)]
    for j in range(n):
        sum = 0
        for k in range(m):
            if allocation[j][k] == 1:
                sum += V[j][k]
        sum_of_valuations_by_bidder[j] = sum
        if sum < val:
            min_bidder = j
            val = sum
    return min_bidder, sum_of_valuations_by_bidder


def calculate_our_algorithm(n, m, R, L, V):
    global MAXITER
    # calculate the optimal matching. maximizing the product of valuations
    v_altered = [[0 for j in range(len(V[0]))] for i in range(len(V))]
    for i in range(len(V)):
        for j in range(len(V[0])):
            v_altered[i][j] = -math.log(V[i][j]) if V[i][j] != 0 else sys.maxsize
    cost_matrix = np.array(v_altered)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Extract the optimal matching
    assignment = [(row, col) for row, col in zip(row_indices, col_indices)]

    # Init an empty allocation
    allocation = [[0 for i in range(m)] for j in range(n)]
    for row, col in assignment:
        allocation[row][col] = 1
    H = [item for (_, item) in assignment]  # Items in optimal matching
    T = [item for item in R if item not in H]  # Items left out of optimal matching
    # Give all items that are not in the matching to the first bidder
    for l in T:
        allocation[0][l] = 1
    change = True  # true if there was a change in the allocation in the last iteration
    bidder_with_min_val = 0
    num_of_iterations = 0
    while change == True:
        change = False
        num_of_iterations += 1
        bidder_with_min_val, sum_of_valuations_by_bidder = find_min_bidder(n, m, allocation, V)

        # This loops checks if it is pays of to give some item to the bidder with minimum valuation
        for other_bidder in range(n):
            if other_bidder == bidder_with_min_val:
                continue
            for item in range(m):
                if (
                    allocation[other_bidder][item] == 1
                ):  # for all items that belong to the other bidder
                    if (
                        V[bidder_with_min_val][item] > 0
                        and sum_of_valuations_by_bidder[other_bidder]
                        - V[other_bidder][item]
                        > sum_of_valuations_by_bidder[bidder_with_min_val]
                    ):
                        # if true, exchange items
                        allocation[other_bidder][item] = 0
                        allocation[bidder_with_min_val][item] = 1
                        change = True
                        break
            if change:
                break

        if change == True:
            continue

        # This loops checks if it is pays of to exchange some item for another item with the bidder with minimum valuation
        for min_bidder_item in range(m):
            if allocation[bidder_with_min_val][min_bidder_item] == 1:
                for other_bidder in range(n):
                    if other_bidder == bidder_with_min_val:
                        continue
                    for other_bidder_item in range(m):
                        if allocation[other_bidder][other_bidder_item] == 1:
                            if (
                                V[bidder_with_min_val][other_bidder_item]
                                - V[bidder_with_min_val][min_bidder_item]
                                > 0
                                and sum_of_valuations_by_bidder[other_bidder]
                                + V[other_bidder][min_bidder_item]
                                - V[other_bidder][other_bidder_item]
                                > sum_of_valuations_by_bidder[bidder_with_min_val]
                            ):
                                # perform exchanging
                                allocation[other_bidder][other_bidder_item] = 0
                                allocation[other_bidder][min_bidder_item] = 1

                                allocation[bidder_with_min_val][other_bidder_item] = 1
                                allocation[bidder_with_min_val][min_bidder_item] = 0
                                change = True
                                break
                    if change:
                        break
            if change:
                break
    if num_of_iterations > MAXITER:
        MAXITER = num_of_iterations
    return sum_of_valuations_by_bidder[bidder_with_min_val]


if __name__ == "__main__":
    worst_approx = 1
    avg_approx = 0
    avgcount = 0

    for i in tqdm(range(NUM_TESTS)):
        our_alg, optimal_max = run_test()
        approx = our_alg / optimal_max
        avg_approx += approx
        if approx < worst_approx:
            worst_approx = approx

    print("worst approximation: ", worst_approx)
    print("average approximation: ", avg_approx / NUM_TESTS)
