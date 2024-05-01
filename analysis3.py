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

def bellman_ford(graph, source):
    # Initialize distances and predecessors dictionaries
    distances = {node: float('inf') for node in graph.nodes()}
    distances[source] = 0
    predecessors = {node: None for node in graph.nodes()}
    
    # Step 1: Relax all edges |V| - 1 times
    for _ in range(len(graph.nodes()) - 1):
        for u, v, w in graph.edges(data='weight'):
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                predecessors[v] = u
    
    # Step 2: Check for negative-weight cycles
    for u, v, w in graph.edges(data='weight'):
        if distances[u] + w < distances[v]:
            print("Graph contains negative weight cycle")
            return None, None
    
    return distances, predecessors

def save_shortest_paths(distances, predecessors, source, filename):
    with open(filename, 'w') as file:
        for node, distance in distances.items():
            path = []
            current = node
            while current is not None:
                path.insert(0, str(current))
                current = predecessors[current]
            file.write(f"Shortest path from {source} to {node}: {' -> '.join(path)}, Distance: {distance}\n")

def run_algorithm_and_measure_time(filename, source_node):
    # Read the graph from the file
    graph = read_graph_from_file(filename)
    
    # Measure the time taken by Bellman-Ford algorithm
    start_time = time.time()
    shortest_distances, predecessors = bellman_ford(graph, source_node)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Execution time for {filename}: {execution_time} seconds")
    return len(graph), execution_time

if __name__ == "__main__":
    graph_sizes = [200, 500, 1000, 1500]
    execution_times = []

    for size in graph_sizes:
        filename = f"graph_{size}.txt"  # Assuming filenames are in the format "graph_size.txt"
        source_node = 0  # Choose a source node
        _, time_taken = run_algorithm_and_measure_time(filename, source_node)
        execution_times.append(time_taken)

    # Plot the graph
    plt.plot(graph_sizes, execution_times, marker='o')
    plt.title('Bellman-Ford Algorithm Execution Time')
    plt.xlabel('Graph Size')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.show()
