from algorithms.utils import Node


def prune(children, frontier, expanded):
    return []


def deterministic_search(problem, initial_state, target_state):
    node = Node([], initial_state)
    frontier = [node]
    expanded = []
    while frontier:
        current_node = frontier[0]
        expanded.append(current_node)
        frontier.pop(0)
        if current_node.state == target_state:
            return current_node.plan
        children = problem.generate_children(current_node.state)
        children = prune(children, frontier, expanded)
        frontier = list(set(frontier).union(set(children)))
    return [False]
