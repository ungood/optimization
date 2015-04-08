#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])
Problem = namedtuple("Problem", ['capacity', 'items'])
Solution = namedtuple("Solution", ['weight', 'value', 'items'])


def parse(input_data):
    """Parse the input"""
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        item = Item(i-1, int(parts[0]), int(parts[1]))
        items.append(item)
        
    return Problem(capacity, items)


def greedy(problem, key):
    sorted_items = sorted(problem.items, key=key)
    
    value = 0
    weight = 0
    taken = []

    for item in sorted_items:
        if weight + item.weight <= problem.capacity:
            taken.append(item)
            value += item.value
            weight += item.weight
            
    return Solution(weight, value, taken)
        
        
def trivial_solution(problem):
    """
    A trivial greedy algorithm for filling the knapsack.
    Takes items in-order until the knapsack is full."""
    return greedy(problem, lambda item: item.index)    

            
def output(problem, solution):
    """Output the solution in the proper format."""
    taken = [0]*len(problem.items)
    
    for item in solution.items:
        taken[item.index] = 1
        
    output_data = str(solution.value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data
    

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    problem = parse(input_data) 
    solution = trivial_solution(problem)
    return output(problem, solution)

import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'

