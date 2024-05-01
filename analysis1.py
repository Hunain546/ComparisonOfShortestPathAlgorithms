import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt

def read_graph_from_file(filename):
    graph = nx.DiGraph()
    with open(filename, 'r') as file:
        for line in file:
            source, target, weight = map(int, line.strip().split())
            graph.add_edge(source, target, weight=weight)
    return graph

def floyd_warshall(matrix):
    num_vertices = len(matrix)
    distance_matrix = matrix.copy()
    
    next_step = np.full((num_vertices, num_vertices), -1)
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j and not np.isinf(distance_matrix[i][j]):
                next_step[i][j] = j
    
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j]:
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                    next_step[i][j] = next_step[i][k]
    
    # Check for negative cycles
    for i in range(num_vertices):
        if distance_matrix[i][i] < 0:
            print("Negative cycle detected")
            return None, None
    
    return distance_matrix, next_step

def reconstruct_shortest_path(next_step, source, target):
    if next_step[source][target] == -1:
        return None  # Target is unreachable from the source
    
    path = []
    node = source
    while node != target:
        path.append(node)
        node = next_step[node][target]
    path.append(target)
    return path

def save_shortest_paths(distance_matrix, next_step, filename):
    num_vertices = len(distance_matrix)
    with open(filename, 'w') as file:
        for i in range(num_vertices):
            for j in range(num_vertices):
                if i != j:
                    path = reconstruct_shortest_path(next_step, i, j)
                    if path is not None:
                        file.write(f"Shortest path from node {i} to node {j}: {' -> '.join(map(str, path))}, Distance: {distance_matrix[i][j]}\n")
                

def graph_to_matrix(graph):
    num_vertices = graph.number_of_nodes()
    matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]
    
    for edge in graph.edges(data=True):
        source, target, weight = edge
        matrix[source][target] = weight['weight']
    
    for i in range(num_vertices):
        matrix[i][i] = 0
    
    return matrix

def run_algorithm_and_measure_time(filename):
    # Read the graph from the file
    graph = read_graph_from_file(filename)
    matrix_representation = graph_to_matrix(graph)
    
    # Measure the time taken by Floyd-Warshall algorithm
    start_time = time.time()
    distance_matrix, next_step = floyd_warshall(matrix_representation)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Execution time for {filename}: {execution_time} seconds")
    return len(matrix_representation), execution_time

if __name__ == "__main__":
    graph_sizes = [200, 500, 1000, 1500]
    execution_times = []

    for size in graph_sizes:
        filename = f"graph_{size}.txt"  # Assuming filenames are in the format "graph_size.txt"
        _, time_taken = run_algorithm_and_measure_time(filename)
        execution_times.append(time_taken)

    # Plot the graph
    plt.plot(graph_sizes, execution_times, marker='o')
    plt.title('Floyd-Warshall Algorithm Execution Time')
    plt.xlabel('Graph Size')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.show()
