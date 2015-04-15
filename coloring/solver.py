#!/usr/bin/python
# -*- coding: utf-8 -*-

# 1/2 = 0.5
from __future__ import division

import argparse
from collections import namedtuple

import logging
logging.basicConfig(format='%(message)s', level=logging.CRITICAL)
log = logging.getLogger(__name__)

Problem = namedtuple('Problem', ['nodes', 'edges'])
Solution = namedtuple('Solution', ['colors', 'optimal'])


class TrivialSolver(object):
    def __init__(self, problem):
        self.colors = problem.nodes
    
    def solve(self):
        return Solution(self.colors, False)


def parse(input_data):
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
        
    nodes = tuple(range(node_count))
    edges = tuple(edges)
    return Problem(nodes, edges)

def output(solution):
    # prepare the solution in the specified output format
    color_count = len(set(solution.colors))
    optimal = 1 if solution.optimal else 0
    
    output_data = '{0} {1}\n'.format(color_count, optimal)
    output_data += ' '.join(map(str, solution.colors))

    return output_data

def solve_it(input_data, solver_class=TrivialSolver):
    problem = parse(input_data)
    solver = solver_class(problem)
    solution = solver.solve()
    return output(solution)

import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'))
    parser.add_argument('--debug', action='store_true')
    
    solvers = parser.add_subparsers(help='Select a solver to use')
    solvers.add_parser('trivial').set_defaults(solver=TrivialSolver)
    
    args = parser.parse_args()
    
    if args.debug:
        log.setLevel(logging.DEBUG)
    
    with args.file as f:
        input_data = ''.join(f.readlines())
        print(solve_it(input_data, args.solver))

