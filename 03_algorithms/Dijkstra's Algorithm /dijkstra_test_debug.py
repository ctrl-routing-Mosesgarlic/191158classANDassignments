"""
Debugging test file for Dijkstra's Algorithm implementations
"""
import os
import sys

print("=== Debugging Dijkstra Test ===")
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))
print("Python path:", sys.path)

# Try importing with different approaches
try:
    print("\nAttempt 1: Direct import")
    from Dijkstras_Algorithm_v1 import dijkstra_simple
    print("Successfully imported Dijkstras_Algorithm_v1")
except ImportError as e:
    print("Import failed:", e)

try:
    print("\nAttempt 2: Absolute path import")
    import sys
    sys.path.append(os.getcwd())
    from Dijkstras_Algorithm_v1 import dijkstra_simple
    print("Successfully imported with absolute path")
except ImportError as e:
    print("Absolute path import failed:", e)

# Test if we can at least open the file
try:
    print("\nAttempt 3: File existence check")
    file_path = os.path.join(os.getcwd(), 'Dijkstras_Algorithm_v1.py')
    with open(file_path, 'r') as f:
        print(f"Successfully opened {file_path}")
except Exception as e:
    print(f"Failed to open file: {e}")

print("\nDebugging complete")
