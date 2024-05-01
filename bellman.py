import networkx as nx

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
    
    # Run Bellman-Ford algorithm
    source_node = 0  # Choose a source node
    shortest_distances, predecessors = bellman_ford(graph, source_node)
    if shortest_distances is None:
        exit()
    # Save the shortest paths to a new file
    output_filename = "bellman.txt"
    save_shortest_paths(shortest_distances, predecessors, source_node, output_filename)
    filename = "bellman_distances.txt"
    save_distances(shortest_distances, filename, source_node)
    print("Shortest distances and paths saved to files")
