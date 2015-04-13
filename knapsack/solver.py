#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import argparse
import random
import heapq
import numpy as np

import logging
logging.basicConfig(format='%(message)s', level=logging.CRITICAL)
log = logging.getLogger(__name__)

def falses(shape):
    return np.zeros(shape, dtype=bool)
    
def trues(shape):
    return np.ones(shape, dtype=bool)

Item = np.dtype([
    ('index',   np.uint),
    ('weight',  np.uint),
    ('value',   np.uint),
    ('density', np.uint)
])

class Problem(object):
    def __init__(self, capacity, items):
        self.capacity = capacity
        self.items = np.array(items, dtype=Item)
        self.N = len(items)
        
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
            items.append((i, weight, value, value/weight))
                
        return Problem(capacity, items)        

class Solution(object):
    def __init__(self, items, selected, optimal=False):
        self.items = items
        self.selected = selected
        self.optimal = optimal
        self.weight = items['weight'][selected].sum()
        self.value = items['value'][selected].sum()
            
    def __str__(self):
        """Output the solution in the proper format."""        
        reordered = np.zeros(len(self.items), dtype=int)
        for i in range(len(self.selected)):
            reordered[self.items[i]['index']-1] = self.selected[i]
        
        optimal = 1 if self.optimal else 0
        output_data = '{0} {1}\n'.format(self.value, optimal)
        output_data += ' '.join(map(str, reordered))
        return output_data
    

class GreedySolver(object):
    def __init__(self, problem, key):
        self.items = sorted(problem.items, key=key)
        self.problem = problem
        
    def __call__(self):
        weight = 0
        selected = falses(self.problem.N)

        for j, item in enumerate(self.items):
            if weight + item['weight'] <= self.problem.capacity:
                selected[j] = True
                weight += item['weight']
            
        return Solution(self.items, selected)


class DynamicSolver(object):
    def __init__(self, problem):
        self.problem = problem
        self.K = problem.capacity
        self.N = problem.N
        self.cache = -1 * np.ones((self.K+1, self.N+1))
        
    def __call__(self):
        self.fill()
        log.debug('%s', self.cache)
        return Solution(self.problem.items, self.backtrack(), optimal=True)
                
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
        

class Branch(object):
    def __init__(self, problem, selected=[], value=0, weight=0):
        self.problem = problem
        self.items = problem.sorted_items
        self.selected = np.array(selected, dtype=bool)
        self.J = len(selected)-1
        self.value = value
        self.weight = weight
        self.leaf = self.J+1 == problem.N
        self.viable = self.weight <= problem.capacity
        self.estimate = self.calc_estimate() if self.viable else -1
            
    def calc_estimate(self):    
        estimate = self.value
        estimated_weight = self.weight
        for i in range(self.J+1, self.problem.N):
            w = self.items[i]['weight']
            v = self.items[i]['value']
            if (w + estimated_weight) > self.problem.capacity:
                take_ratio = (self.problem.capacity - estimated_weight) / w
                partial_value = v * take_ratio
                return estimate + partial_value
            else:
                estimate += v
                estimated_weight += w
        return estimate
        
    def __cmp__(self, other):
        # heapq keeps the smallest item at the top, and I want to evaluate best-first,
        # so I need to reverse the compare.
        return cmp(other.value, self.value)
        
    def left(self):
        if self.leaf:
            return None
        
        selected = np.append(self.selected, [False])
        return Branch(self.problem, selected, self.value, self.weight)
    
    def right(self):
        if self.leaf:
            return None
        
        selected = np.append(self.selected, [True])
        next_item = self.items[self.J+1]
        return Branch(self.problem, selected,
            value=self.value + next_item['value'],
            weight=self.weight + next_item['weight'])
                
class BranchAndBoundSolver(object):
    def __init__(self, problem):
        problem.sorted_items = problem.items.copy()
        problem.sorted_items.sort(order=['density'])
        self.problem = problem
        
    def __call__(self):
        root = Branch(self.problem)
        queue = [root]
        best = root
        
        visited = 0
        pruned = 0
        skipped = 0
                
        while len(queue) > 0:
            visited += 1
            current = heapq.heappop(queue)
            
            if current.estimate < best.value:
                log.debug('%d is less than %d. Pruned.', current.estimate, best.value)
                pruned += 1
                continue
                    
            if current.value > best.value:
                log.debug('%d is new best.', current.value)
                best  = current
                
            if not current.leaf:
                left = current.left()
                if left.viable:
                    heapq.heappush(queue, current.left())
                right = current.right()
                if right.viable:
                    heapq.heappush(queue, current.right())
                        
        log.debug('Visited: %d, Pruned: %d, Skipped: %d', visited, pruned, skipped)
        return Solution(best.items, best.selected)
    
    
class GeneticSolver(object):
    def __init__(self, problem, population_size, iterations):
        self.population = np.random.random_integers(0, 1, (population_size, problem.N))
                
        for i in range(iterations):
            weights = np.dot(population, problem.items['weight'])
            viable = weights <= problem.capacity
            population = population[viable]
        # WIP


class OpenOptSolver(object):
    """
    This is 'cheating' by using OpenOpt to solve the knapsack problem.  I am not using this solver for my assignments,
    but rather to compare running time and solutions against a 'known' solution."""
    def __init__(self, problem):
        self.items = problem.items
        self.constraints = lambda items: items['weight'] < problem.capacity
        
    def __call__(self):
        from openopt import *
        p = KSP('value', self.items, constraints = self.constraints)
        r = p.solve('glpk', iprint=0)
        print(r)


def solve_it(input_data):
    problem = Problem.parse(input_data)
#    solver = GreedySolver(problem, lambda item: item['index'])
#    solver = BranchAndBoundSolver(problem)
    solver = OpenOptSolver(problem)
    return str(solver())

import sys, os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'))
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    if args.debug:
        log.setLevel(logging.DEBUG)
    
    with args.file as f:
        input_data = ''.join(f.readlines())
        print solve_it(input_data)