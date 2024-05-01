import random
import networkx as nx
import matplotlib.pyplot as plt

def generate_weighted_dag(num_vertices):
    """
    Generates a random DAG with the specified number of vertices and 
    includes negative weights.
    """

    G = nx.DiGraph()  # Create an empty directed graph

    # Add vertices
    for i in range(num_vertices):
        G.add_node(i)

    # Add edges while ensuring no cycles
    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):  # Only add edges forward
            if random.random() < 0.5:  # Add edges with some probability
                weight = random.randint(-10, 10)  # Include negative weights
                G.add_edge(u, v, weight=weight)
    
    # Save the graph to a text file
    with open("graph_dag.txt", 'w') as file:
        for edge in G.edges():
            source, target = edge
            weight = G[source][target]['weight']
            file.write(f"{source} {target} {weight}\n")

    return G

# Example usage
num_vertices = 10
my_dag = generate_weighted_dag(num_vertices)

# Visualize
pos = nx.spring_layout(my_dag, seed=42)
nx.draw(my_dag, pos, with_labels=True, node_size=1000, font_size=20, node_color='lightblue', edge_color='gray')
labels = nx.get_edge_attributes(my_dag, 'weight')
nx.draw_networkx_edge_labels(my_dag, pos, edge_labels=labels)
plt.title("Random Directed Acyclic Graph with Weights")
plt.show()

