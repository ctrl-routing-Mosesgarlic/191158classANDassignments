"""
Test file for Dijkstra's Algorithm implementations
"""
import os

print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))
print("\nTesting Dijkstra implementations...")

# Sample graph
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

# Test Version 1
print("\n=== Testing Version 1 ===")
try:
    from Dijkstras_Algorithm_v1 import dijkstra_simple
    distance, path = dijkstra_simple(graph, 'A', 'D')
    print(f"Distance: {distance}")
    print(f"Path: {' -> '.join(path)}")
except ImportError as e:
    print(f"Could not import v1: {e}")
except Exception as e:
    print(f"Error running v1: {e}")

# Test Version 2
print("\n=== Testing Version 2 ===")
try:
    from Dijkstras_Algorithm_v2 import DijkstraIntermediate
    dijkstra = DijkstraIntermediate(graph)
    path, distance = dijkstra.find_shortest_path('A', 'D')
    print(f"Distance: {distance}")
    print(f"Path: {' -> '.join(path)}")
except ImportError as e:
    print(f"Could not import v2: {e}")
except Exception as e:
    print(f"Error running v2: {e}")

# Test Version 3
print("\n=== Testing Version 3 ===\n")
try:
    from Dijkstras_Algorithm_v3 import DijkstraAdvanced
    test_graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    # Test basic functionality
    print("Testing basic path finding:")
    advanced = DijkstraAdvanced(test_graph)
    distances, previous, analysis = advanced.dijkstra_with_analysis('A', 'D')
    path = advanced.reconstruct_path('A', 'D', previous)
    print(f"Distance from A to D: {distances['D']}")
    print(f"Path: {' -> '.join(path)}")
    
    # Test analysis features
    print("\nTesting analysis features:")
    print(f"Nodes processed: {analysis['nodes_processed']}")
    print(f"Edges relaxed: {analysis['edges_relaxed']}")
    print(f"Completion: {analysis['completion_percentage']:.1f}%")
    
except ImportError as e:
    print(f"Error testing Version 3: {e}")

print("\nTesting complete")
