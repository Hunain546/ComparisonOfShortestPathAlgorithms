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

def dijkstra(graph):
    """Dijkstra's algorithm for All-Pairs Shortest Paths (APSP) problem."""
    all_shortest_distances = {}
    
    for source in graph.nodes():
        # Initialize distances, predecessors, and visited sets for the current source
        distances = {node: float('inf') for node in graph.nodes()}
        distances[source] = 0
        predecessors = {node: None for node in graph.nodes()}
        visited = set()
        
        # Priority queue for Dijkstra's algorithm
        pq = [(0, source)]
        
        while pq:
            dist_to_current, current = heapq.heappop(pq)
            
            # If this node has already been visited with a shorter distance, skip it
            if current in visited:
                continue
            
            # Mark the current node as visited
            visited.add(current)
            
            # Explore neighbors
            for neighbor, attrs in graph[current].items():
                if neighbor in visited:
                    continue  # Skip already visited neighbors
                distance_to_neighbor = dist_to_current + attrs['weight']
                if distance_to_neighbor < distances[neighbor]:
                    distances[neighbor] = distance_to_neighbor
                    predecessors[neighbor] = current
                    heapq.heappush(pq, (distance_to_neighbor, neighbor))
        
        # Store the shortest distances for the current source
        all_shortest_distances[source] = distances
    
    return all_shortest_distances



def bellman_ford(graph):
    """Bellman-Ford algorithm for All-Pairs Shortest Paths (APSP) problem."""
    num_nodes = len(graph.nodes())
    all_shortest_distances = {}
    
    # Initialize distances and predecessors dictionaries for all source nodes
    distances = {source: {node: float('inf') for node in graph.nodes()} for source in graph.nodes()}
    predecessors = {source: {node: None for node in graph.nodes()} for source in graph.nodes()}
    
    # Run Bellman-Ford for each vertex as the source node
    for source in graph.nodes():
        # Set the distance from the source to itself to be 0
        distances[source][source] = 0
        
        # Relax edges repeatedly for |V| - 1 iterations
        for _ in range(num_nodes - 1):
            # Keep track of whether any distances were updated in this iteration
            updated = False
            
            # Relax edges only for vertices whose shortest paths were updated in the previous iteration
            for u, v, attrs in graph.edges(data=True):
                w = attrs['weight']
                if distances[source][u] + w < distances[source][v]:
                    distances[source][v] = distances[source][u] + w
                    predecessors[source][v] = u
                    updated = True
            
            # If no distances were updated in this iteration, terminate early
            if not updated:
                break
        
        # Check for negative-weight cycles
        for u, v, attrs in graph.edges(data=True):
            w = attrs['weight']
            if distances[source][u] + w < distances[source][v]:
                print("Graph contains negative weight cycle")
                return None
        
        # Store the shortest distances for the current source
        all_shortest_distances[source] = distances[source]
    
    return all_shortest_distances






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

def run_algorithm_and_measure_time(algorithm, graph):
    start_time = time.time()
    if algorithm == "Dijkstra":
        shortest_distances = dijkstra(graph)
    elif algorithm == "Bellman-Ford":
        shortest_distances = bellman_ford(graph)
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
    graph_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    algorithms = ["Dijkstra","Floyd-Warshall"]
    execution_times = {algorithm: [] for algorithm in algorithms}

    for size in graph_sizes:
        filename = f"graph_100_{size}.txt"  # Assuming filenames are in the format "graph_size.txt"
        graph = read_graph_from_file(filename)
        source_node = 0  # Choose a source node

        for algorithm in algorithms:
            time_taken = run_algorithm_and_measure_time(algorithm, graph)
            print(f"Execution time for {algorithm} on {filename}: {time_taken} seconds")
            execution_times[algorithm].append(time_taken)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        plt.plot(graph_sizes, execution_times[algorithm], marker='o', label=algorithm)

    plt.title('Comparison As All Pair Shortest Path Algorithms')
    plt.xlabel('Graph Density (Probability of Edge Creation)')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.show()

