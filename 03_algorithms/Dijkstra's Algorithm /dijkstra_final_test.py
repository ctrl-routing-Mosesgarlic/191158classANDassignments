"""
Final test file for Dijkstra implementations
"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())
print(f"Python path: {sys.path}")

# Test Version 1
print("\n=== Testing Version 1 ===")
try:
    import Dijkstras_Algorithm_v1
    print("Successfully imported v1")
    
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    
    distance, path = Dijkstras_Algorithm_v1.dijkstra_simple(graph, 'A', 'D')
    print(f"Distance: {distance}")
    print(f"Path: {' -> '.join(path)}")
except Exception as e:
    print(f"Error: {str(e)}")
    print(f"Full error: {repr(e)}")

print("\nTest complete")
