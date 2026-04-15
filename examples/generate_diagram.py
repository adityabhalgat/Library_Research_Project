"""Generate a sample architecture diagram for testing."""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx

def generate_diagram():
    try:
        # Create a sample microservices architecture graph
        G = nx.DiGraph()
        G.add_edge("User Client", "API Gateway")
        G.add_edge("API Gateway", "Auth Service")
        G.add_edge("API Gateway", "Product Service")
        G.add_edge("API Gateway", "Order Service")
        G.add_edge("Product Service", "Product DB")
        G.add_edge("Order Service", "Order DB")
        G.add_edge("Order Service", "Payment Gateway")
        
        # Draw the graph
        plt.figure(figsize=(10, 8))
        pos = nx.circular_layout(G)
        nx.draw(
            G, pos, 
            with_labels=True, 
            node_color='lightblue', 
            node_size=3000, 
            font_size=10, 
            font_weight='bold',
            arrowsize=20,
            edge_color='gray'
        )
        plt.title("Sample E-Commerce Architecture")
        
        # Save to file
        output_path = "examples/sample_inputs/architecture_diagram.png"
        plt.savefig(output_path)
        print(f"Successfully generated: {output_path}")
        
    except Exception as e:
        print(f"Error generating diagram: {e}")

if __name__ == "__main__":
    generate_diagram()
