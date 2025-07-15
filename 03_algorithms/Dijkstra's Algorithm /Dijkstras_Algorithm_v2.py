# ============================================================================
# VERSION 2: INTERMEDIATE VERSION WITH OPTIMIZATIONS
# Using priority queue for better performance
# ============================================================================

import heapq

print("\n\n" + "=" * 60)
print("VERSION 2: INTERMEDIATE VERSION (High School Level)")
print("=" * 60)

class DijkstraIntermediate:
    """
    Intermediate version with priority queue optimization
    
    This version introduces:
    1. Priority queue (heap) for efficient minimum distance selection
    2. Better data structures
    3. Path reconstruction
    4. Multiple utility functions
    """
    
    def __init__(self, graph):
        """
        Initialize with a graph
        Graph format: {node: {neighbor: weight, ...}, ...}
        """
        self.graph = graph
        self.nodes = set(graph.keys())
        
        # Add all neighbors to nodes set (in case they're not in keys)
        for node in graph:
            for neighbor in graph[node]:
                self.nodes.add(neighbor)
    
    def dijkstra(self, start, end=None):
        """
        Find shortest paths from start to all nodes (or specific end node)
        
        Uses a priority queue (heap) to always process the node with
        minimum distance first, which makes the algorithm more efficient.
        """
        
        print(f"Starting Dijkstra from node: {start}")
        
        # Initialize distances and previous nodes
        distances = {node: float('inf') for node in self.nodes}
        previous = {node: None for node in self.nodes}
        distances[start] = 0
        
        # Priority queue: (distance, node)
        # heapq always gives us the smallest distance first
        pq = [(0, start)]
        visited = set()
        
        print(f"Initial state: {len(self.nodes)} nodes to explore")
        
        while pq:
            # Get the node with minimum distance
            current_distance, current_node = heapq.heappop(pq)
            
            # Skip if we've already processed this node
            if current_node in visited:
                continue
            
            # Mark as visited
            visited.add(current_node)
            print(f"Processing node {current_node} (distance: {current_distance})")
            
            # If we only want path to specific end node and we found it
            if end and current_node == end:
                break
            
            # Explore all neighbors
            if current_node in self.graph:
                for neighbor, weight in self.graph[current_node].items():
                    if neighbor not in visited:
                        # Calculate new distance through current node
                        new_distance = current_distance + weight
                        
                        # If we found a shorter path, update it
                        if new_distance < distances[neighbor]:
                            distances[neighbor] = new_distance
                            previous[neighbor] = current_node
                            heapq.heappush(pq, (new_distance, neighbor))
                            print(f"  Updated {neighbor}: distance = {new_distance}")
        
        return distances, previous
    
    def get_path(self, start, end, previous):
        """
        Reconstruct the shortest path from start to end
        """
        path = []
        current = end
        
        # Trace back from end to start using previous nodes
        while current is not None:
            path.append(current)
            current = previous[current]
        
        path.reverse()
        
        # Check if path is valid (should start with start node)
        if path[0] != start:
            return None  # No path exists
        
        return path
    
    def find_shortest_path(self, start, end):
        """
        Find the shortest path and distance between start and end
        """
        distances, previous = self.dijkstra(start, end)
        path = self.get_path(start, end, previous)
        
        if path is None:
            return None, float('inf')
        
        return path, distances[end]
    
    def print_all_shortest_paths(self, start):
        """
        Print shortest paths from start to all other nodes
        """
        distances, previous = self.dijkstra(start)
        
        print(f"\nShortest paths from {start}:")
        print("-" * 40)
        
        for node in sorted(self.nodes):
            if node != start:
                path = self.get_path(start, node, previous)
                if path:
                    print(f"To {node}: {' -> '.join(path)} (distance: {distances[node]})")
                else:
                    print(f"To {node}: No path exists")

# Example 2: More complex network
print("\nExample 2: Computer Network Routing")
print("-" * 50)

# Represents a computer network where:
# - Nodes are computers/routers
# - Edges are network connections with latency/cost
network_graph = {
    'Router1': {'Router2': 1, 'Router3': 4, 'Server1': 7},
    'Router2': {'Router1': 1, 'Router3': 2, 'Router4': 3},
    'Router3': {'Router1': 4, 'Router2': 2, 'Router4': 1, 'Server2': 5},
    'Router4': {'Router2': 3, 'Router3': 1, 'Server2': 2, 'Server3': 6},
    'Server1': {'Router1': 7, 'Server2': 3},
    'Server2': {'Router3': 5, 'Router4': 2, 'Server1': 3, 'Server3': 1},
    'Server3': {'Router4': 6, 'Server2': 1}
}

dijkstra_net = DijkstraIntermediate(network_graph)

# Find shortest path between specific nodes
path, distance = dijkstra_net.find_shortest_path('Router1', 'Server3')
print(f"Shortest path from Router1 to Server3: {' -> '.join(path)}")
print(f"Total latency: {distance}")

# Show all shortest paths from Router1
dijkstra_net.print_all_shortest_paths('Router1')
