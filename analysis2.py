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
    # Initialize distances, predecessors, and visited sets
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
    
    # Measure the time taken by Dijkstra's algorithm
    start_time = time.time()
    shortest_distances, predecessors = dijkstra(graph, source_node)
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
    plt.title('Dijkstra\'s Algorithm Execution Time')
    plt.xlabel('Graph Size')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.show()
