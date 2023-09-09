# Max-Min Welfare Problem Algorithm

This repository contains the code implementation of our solution for the Max-Min welfare problem. We've designed this based on the local search methodology referenced from [garg2023approximating](https://arxiv.org/abs/2211.03883).
#### Contributors
- Hen Kas-Sharir (Technion)
- Amit Ganz (Technion)
## Overview

The algorithm is divided into two main steps:

1. **Initial Solution Generation**: Find a starting point that serves as our initial solution.
2. **Refinement**: Improve the initial solution to get closer to the optimal Max-Min value using local search.

## File Structure and Explanation

### `sol.py`

This file encompasses our entire solution for the problem. Here's what you can find inside:

- **`run_test()`**: Executes a single test instance.
- **`calculate_optimal_by_range()`**: Computes the optimal Max-Min solution by evaluating within a specified range. This is done using a naive algorithm to go over all possible allocations.
- **`calculate_our_algorithm()`**: Implements our proposed algorithm to find a solution to the Max-Min problem.

## Configuration Parameters

In our solution, we have set several configuration parameters to control various aspects of the tests. They are:

- **`MAX_BIDDERS`**: This represents the maximum number of bidders possible. During actual tests, the number of bidders is randomized from a range between 1 and `MAX_BIDDERS`.

- **`MAX_ITEMS`**: The maximum number of items is set to twice the value of `MAX_BIDDERS`. The exact number of items used in tests is randomized.

- **`MAX_VALUATION`**: This represents the highest valuation a bidder can have for an item. The actual valuation for tests is randomly chosen up to this maximum.

- **`NUM_TESTS`**: This parameter sets the number of tests to run.


### How to Use

1. To the entire test set run in your command line inside the directory with sol.py:
   ```bash
   python3 sol.py
## Prerequisites

Before running the code, ensure that you have the following setup:

- **Python 3**: The code is written in Python 3 and is required for execution.
- **Required Libraries**: Install the necessary libraries using the following:
  ```bash
  pip3 install numpy tqdm scipy
