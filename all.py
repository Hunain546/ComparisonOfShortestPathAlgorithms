import networkx as nx
import heapq
import time
import matplotlib.pyplot as plt

def read_graph_from_file(filename):
    graph = nx.DiGraph()
    with open(filename, 'r') as file:
        for line in file:
            source, target, weight = map(int, line.strip().split())
            graph.add_edge(source, target, weight=weight)
    return graph

def dijkstra(graph, source):
    # Initialize distances and predecessors dictionaries
    distances = {node: float('inf') for node in graph.nodes()}
    distances[source] = 0
    predecessors = {node: None for node in graph.nodes()}
    
    # Priority queue for Dijkstra's algorithm
    pq = [(0, source)]
    
    while pq:
        dist_to_current, current = heapq.heappop(pq)
        
        # If this node has already been visited with a shorter distance, skip it
        if dist_to_current > distances[current]:
            continue
        
        # Explore neighbors
        for neighbor, attrs in graph[current].items():
            distance_to_neighbor = dist_to_current + attrs['weight']
            if distance_to_neighbor < distances[neighbor]:
                distances[neighbor] = distance_to_neighbor
                predecessors[neighbor] = current
                heapq.heappush(pq, (distance_to_neighbor, neighbor))
    
    return distances, predecessors

def bellman_ford(graph, source):
    # Initialize distances and predecessors dictionaries
    distances = {node: float('inf') for node in graph.nodes()}
    distances[source] = 0
    predecessors = {node: None for node in graph.nodes()}
    
    # Keep track of vertices whose distances have been updated
    updated_vertices = {source}
    
    # Step 1: Relax edges repeatedly until no update or |V| - 1 iterations
    for _ in range(len(graph.nodes()) - 1):
        # Create a new set to store vertices to be updated in the next iteration
        next_updated_vertices = set()
        
        # Iterate over edges only from vertices whose distances have been updated
        for u in updated_vertices:
            for v, attrs in graph[u].items():
                w = attrs['weight']
                if distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    predecessors[v] = u
                    # Update the set of vertices to be updated in the next iteration
                    next_updated_vertices.add(v)
        
        # Update the set of updated vertices for the next iteration
        updated_vertices = next_updated_vertices
        
        # If no distance is updated, terminate early
        if not updated_vertices:
            break
    
    # Step 2: Check for negative-weight cycles
    for u in graph.nodes():
        for v, attrs in graph[u].items():
            w = attrs['weight']
            if distances[u] + w < distances[v]:
                print("Graph contains negative weight cycle")
                return None, None
    
    return distances, predecessors

def floyd_warshall(matrix):
    num_vertices = len(matrix)
    distance_matrix = matrix.copy()
    
    next_step = [[j for j in range(num_vertices)] for _ in range(num_vertices)]
    
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

def graph_to_matrix(graph):
    num_vertices = graph.number_of_nodes()
    matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]
    
    for edge in graph.edges(data=True):
        source, target, weight = edge
        matrix[source][target] = weight['weight']
    
    for i in range(num_vertices):
        matrix[i][i] = 0
    
    return matrix

def save_shortest_paths(distances, predecessors, source, filename):
    with open(filename, 'w') as file:
        for node, distance in distances.items():
            path = []
            current = node
            while current is not None:
                path.insert(0, str(current))
                current = predecessors[current]
            file.write(f"Shortest path from {source} to {node}: {' -> '.join(path)}, Distance: {distance}\n")

def run_algorithm_and_measure_time(algorithm, graph, source_node):
    start_time = time.time()
    if algorithm == "Dijkstra":
        shortest_distances, predecessors = dijkstra(graph, source_node)
    elif algorithm == "Bellman-Ford":
        shortest_distances, predecessors = bellman_ford(graph, source_node)
    elif algorithm == "Floyd-Warshall":
        matrix_representation = graph_to_matrix(graph)
        shortest_distances, _ = floyd_warshall(matrix_representation)
    else:
        print("Unknown algorithm")
        return None
    end_time = time.time()
    
    execution_time = end_time - start_time
    return execution_time


if __name__ == "__main__":
    # graph_sizes = [200, 500, 1000,1500]
    graph_sizes = [0.2, 0.4, 0.6, 0.8]
    algorithms = ["Dijkstra", "Bellman-Ford", "Floyd-Warshall"]
    execution_times = {algorithm: [] for algorithm in algorithms}

    for size in graph_sizes:
        filename = f"graph_{size}.txt"  # Assuming filenames are in the format "graph_size.txt"
        graph = read_graph_from_file(filename)
        source_node = 0  # Choose a source node

        for algorithm in algorithms:
            time_taken = run_algorithm_and_measure_time(algorithm, graph, source_node)
            print(f"Execution time for {algorithm} on {filename}: {time_taken} seconds")
            execution_times[algorithm].append(time_taken)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        plt.plot(graph_sizes, execution_times[algorithm], marker='o', label=algorithm)

    plt.title('Comparison of Shortest Path Algorithms')
    plt.xlabel('Graph Density')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.show()
