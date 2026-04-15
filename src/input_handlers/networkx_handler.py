"""NetworkX graph handler for architecture diagrams."""

import base64
import io
import pickle
from pathlib import Path
from typing import Any


class NetworkXHandler:
    """Handler for NetworkX graphs - converts to image for LLM processing."""
    
    def __init__(self, graph_or_path: Any = None):
        """Initialize with a NetworkX graph or path to pickle file.
        
        Args:
            graph_or_path: Either a NetworkX graph object or path to pickle file
        """
        self.graph = None
        
        if graph_or_path is not None:
            if isinstance(graph_or_path, (str, Path)):
                self.load_from_pickle(str(graph_or_path))
            else:
                # Assume it's a NetworkX graph
                self.graph = graph_or_path
    
    def load_from_pickle(self, path: str) -> None:
        """Load a NetworkX graph from a pickle file.
        
        Args:
            path: Path to the pickle file
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        
        with open(file_path, 'rb') as f:
            self.graph = pickle.load(f)
    
    def set_graph(self, graph: Any) -> None:
        """Set the NetworkX graph directly.
        
        Args:
            graph: NetworkX graph object
        """
        self.graph = graph
    
    def to_image_base64(
        self,
        figsize: tuple[int, int] = (12, 8),
        title: str = "Architecture Diagram",
        with_labels: bool = True,
        node_size: int = 2000,
        node_color: str = "lightblue",
        font_size: int = 10,
        edge_color: str = "gray",
        layout: str = "spring"
    ) -> tuple[str, str]:
        """Convert the NetworkX graph to a base64 encoded PNG image.
        
        Args:
            figsize: Figure size in inches
            title: Title for the diagram
            with_labels: Whether to show node labels
            node_size: Size of nodes
            node_color: Color of nodes
            font_size: Font size for labels
            edge_color: Color of edges
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'shell')
            
        Returns:
            Tuple of (base64_string, media_type)
            
        Raises:
            ValueError: If no graph is loaded
        """
        if self.graph is None:
            raise ValueError("No graph loaded. Use load_from_pickle() or set_graph() first.")
        
        # Import here to avoid issues if matplotlib/networkx not installed
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        layout_funcs = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'shell': nx.shell_layout,
            'random': nx.random_layout
        }
        layout_func = layout_funcs.get(layout, nx.spring_layout)
        
        try:
            pos = layout_func(self.graph)
        except Exception:
            # Fallback to spring layout if chosen layout fails
            pos = nx.spring_layout(self.graph)
        
        # Draw graph
        nx.draw(
            self.graph,
            pos,
            ax=ax,
            with_labels=with_labels,
            node_size=node_size,
            node_color=node_color,
            font_size=font_size,
            edge_color=edge_color,
            arrows=True if self.graph.is_directed() else False
        )
        
        # Draw edge labels if they exist
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=8)
        
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)
        
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_str, "image/png"
    
    def get_graph_info(self) -> dict:
        """Get information about the loaded graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if self.graph is None:
            return {"loaded": False}
        
        import networkx as nx
        
        return {
            "loaded": True,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "node_list": list(self.graph.nodes()),
            "density": nx.density(self.graph)
        }
