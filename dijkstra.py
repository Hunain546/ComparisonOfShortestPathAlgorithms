import networkx as nx
import heapq

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

def save_distances(distances, filename, source):
    with open(filename, 'w') as file:
        for node, distance in distances.items():
            file.write(f"Shortest distance from {source} to {node}: {distance}\n")


if __name__ == "__main__":
    # Read the graph from the file
    filename = "graph.txt"
    graph = read_graph_from_file(filename)
    
    # Run Dijkstra's algorithm
    source_node = 0  # Choose a source node
    shortest_distances, predecessors = dijkstra(graph, source_node)
    
    # Save the shortest paths to a new file
    output_filename = "dijkstra.txt"
    save_shortest_paths(shortest_distances, predecessors, source_node, output_filename)
    filename = "dijkstra_distances.txt"
    save_distances(shortest_distances, filename, source_node)
    print("Shortest distances and paths saved to files")

    

