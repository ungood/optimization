#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import numpy as np

def falses(shape):
    return np.zeros(shape, dtype=bool)

Item = np.dtype([
    ('index',np.uint),
    ('weight',np.uint),
    ('value',np.uint),
])

class Problem(object):
    def __init__(self, capacity, items):
        self.capacity = capacity
        self.items = np.array(items, dtype=Item)

    def sum(self, selected, record):
        return self.items[record][selected].sum()
        
    def value(self, index):
        return self.items['value'][index-1]
        
    def weight(self, index):
        return self.items['weight'][index-1]

    @staticmethod
    def parse(input_data):
        lines = input_data.split('\n')

        firstLine = lines[0].split()
        item_count = int(firstLine[0])
        capacity = int(firstLine[1])

        items = []

        for i in range(1, item_count+1):
            line = lines[i]
            parts = line.split()
            value = int(parts[0])
            weight = int(parts[1])
            items.append((i, weight, value))
                
        return Problem(capacity, items)        

class Solution(object):
    def __init__(self, problem, selected, optimal=False):
        self.problem = problem
        self.optimal = optimal
        self.selected = selected
        self.weight = problem.sum(selected, 'weight')
        self.value = problem.sum(selected, 'value')
        self.viable = self.weight <= self.problem.capacity
        
    def output(self):
        """Output the solution in the proper format."""        
        optimal = 1 if self.optimal else 0
        selected = np.array(self.selected, dtype=int)
        output_data = '{0} {1}\n'.format(self.value, optimal)
        output_data += ' '.join(map(str, selected))
        return output_data
    

class GreedySolver(object):
    def __init__(self, problem, key):
        self.items = sorted(problem.items, key=key)
        self.problem = problem
        
    def __call__(self):
        weight = 0
        selected = falses(len(self.items))

        for j, item in enumerate(self.items):
            if weight + item['weight'] <= self.problem.capacity:
                selected[j] = True
                weight += item['weight']
            
        return Solution(self.problem, selected)


class DynamicSolver(object):
    def __init__(self, problem):
        self.problem = problem
        self.K = problem.capacity
        self.N = len(problem.items)
        self.cache = -1 * np.ones((self.K+1, self.N+1))
        
    def __call__(self):
        self.fill()        
        return Solution(self.problem, self.backtrack(), optimal=True)
                
    def fill(self):
        for j in range(0, self.N+1):
            for k in range(0, self.K+1):
                value = self.compute(k, j)
                self.cache[k, j] = value
                
    def compute(self, k, j):
        if j == 0:
            return 0
        weight = self.problem.weight(j)
        value = self.problem.value(j)
        if weight <= k:
            take_value = value + self.cache[k-weight, j-1]
            leave_value = self.cache[k, j-1]
            return max(take_value, leave_value)
        else:
            return self.cache[k, j-1]
        
    def backtrack(self):
        selected = falses(self.N)
        j = self.N
        k = self.K
        for j in range(self.N, 0, -1):
            value = self.cache[k, j]
            prev = self.cache[k, j-1]
            if value != prev:
                selected[j-1] = True
                k -= self.problem.weight(j)
                
        return selected
    
    
class GeneticSolver(object):
    def __init__(self, problem, population_size, iterations):
        self.population = np.random.random_integers(0, 1, (population_size, len(problem.items)))
                
        for i in range(iterations):
            print(population)
            weights = np.dot(population, problem.items['weight'])
            viable = weights <= problem.capacity
            population = population[viable]
        # WIP


def solve_it(input_data):
    problem = Problem.parse(input_data)
#    solver = GreedySolver(problem, lambda item: item['index'])
    solver = DynamicSolver(problem)
    solution = solver()
    return solution.output()

def solve_file(filename):
    with open(filename, 'r') as file:
        input_data = ''.join(file.readlines())
        return solve_it(input_data)

import sys, os

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_file = sys.argv[1].strip()
        if os.path.isfile(input_file):
            print(solve_file(input_file))
        elif os.path.isdir(input_file):
            for file in os.listdir(input_file):
                filename = os.path.join(input_file, file)
                print(file)
                print(solve_file(filename))
        else:
            print('{0} is not a valid file or directory.'.format(input_file))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

