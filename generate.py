import networkx as nx
import random
import matplotlib.pyplot as plt

def generate_and_save_graph(vertices, filename):
    # Generate a graph
    graph = nx.fast_gnp_random_graph(vertices, p=1, directed=True)


    # Assign random weights to edges
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['weight'] = random.randint(1, 10)  # Random weight between 1 and 10

    # for edge in graph.edges():
    #     print(edge)

    # Save the graph to a text file
    with open(filename, 'w') as file:
        for edge in graph.edges():
            source, target = edge
            weight = graph[source][target]['weight']
            file.write(f"{source} {target} {weight}\n")
    
    return graph

# Example usage
vertices = 200
filename = "graph.txt"
graph = generate_and_save_graph(vertices, filename)
print("done")

# # Visualize the graph
# plt.figure(figsize=(12, 6))
# pos = nx.spring_layout(graph, seed=42)
# nx.draw(graph, pos, with_labels=True, node_size=1000, font_size=20, node_color='lightblue', edge_color='gray')
# labels = nx.get_edge_attributes(graph, 'weight')
# nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
# plt.title("Random Directed Graph with Weights")
# plt.show()
