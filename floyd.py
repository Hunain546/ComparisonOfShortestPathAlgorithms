import numpy as np
import networkx as nx

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

def save_distances(distances, filename, source):
    with open(filename, 'w') as file:
        for target, distance in enumerate(distances):
            if distance != float('inf'):  # Only save reachable nodes
                file.write(f"Shortest distance from {source} to {target}: {distance}\n")

                

def graph_to_matrix(graph):
    num_vertices = graph.number_of_nodes()
    matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]
    
    for edge in graph.edges(data=True):
        source, target, weight = edge
        matrix[source][target] = weight['weight']
    
    for i in range(num_vertices):
        matrix[i][i] = 0
    
    return matrix
                        
if __name__ == "__main__":
    # Read the graph from the file
    filename = "graph_dag.txt"
    graph = read_graph_from_file(filename)
    matrix_representation = graph_to_matrix(graph)
    print(matrix_representation)
    
    # Run Floyd-Warshall algorithm
    distance_matrix, next_step = floyd_warshall(matrix_representation)
    if distance_matrix is not None:
        # Save the shortest paths to a new file
        output_filename = "floyd.txt"
        save_shortest_paths(distance_matrix, next_step, output_filename)
        filename = "floyd_distances.txt"
        save_distances(distance_matrix[0], filename, 0)
        print("Shortest distances and paths saved to files")
