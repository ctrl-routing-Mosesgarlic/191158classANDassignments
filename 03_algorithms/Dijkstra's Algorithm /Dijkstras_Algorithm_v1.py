# ============================================================================
# VERSION 1: SIMPLE VERSION FOR BEGINNERS
# Think of it like finding the shortest path in a city map
# ============================================================================

print("\n" + "=" * 60)
print("VERSION 1: SIMPLE VERSION (Junior School Level)")
print("=" * 60)

def dijkstra_simple(graph, start, end):
    """
    Simple version of Dijkstra's algorithm
    
    Think of this like finding the shortest route between two cities:
    - You start at one city (start) Nairobi
    - You want to reach another city (end) Nakuru
    - You have distances between cities (graph)
    - You want to find the shortest total distance
    """
    
    # Step 1: Create a table to store the shortest distance to each city
    # Initially, we don't know the distance to any city, so we set them to "infinity"
    # (we use a very large number to represent infinity)
    distances = {}
    for city in graph:
        distances[city] = float('inf')  # infinity means "we don't know yet"
    
    # Step 2: We know the distance from start city to itself is 0
    distances[start] = 0
    
    # Step 3: Keep track of cities we've already visited
    visited = set()
    
    # Step 4: Keep track of the path we took to get to each city
    previous = {}
    
    print(f"Starting journey from {start} to {end}")
    print(f"Initial distances: {distances}")
    
    # Step 5: Keep exploring until we've visited all cities
    while len(visited) < len(graph):
        
        # Find the unvisited city with the shortest known distance
        current_city = None
        shortest_distance = float('inf')
        
        for city in distances:
            if city not in visited and distances[city] < shortest_distance:
                shortest_distance = distances[city]
                current_city = city
        
        # If we can't find any unvisited city, we're done
        if current_city is None:
            break
        
        print(f"\\nVisiting city: {current_city} (distance: {distances[current_city]})")
        
        # Mark this city as visited
        visited.add(current_city)
        
        # If we reached our destination, we can stop
        if current_city == end:
            break
        
        # Look at all neighboring cities
        for neighbor, road_distance in graph[current_city].items():
            if neighbor not in visited:
                # Calculate the total distance if we go through current_city
                new_distance = distances[current_city] + road_distance
                
                # If this path is shorter than what we knew before, update it
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_city
                    print(f"  Found shorter path to {neighbor}: {new_distance}")
    
    # Step 6: Reconstruct the shortest path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous.get(current)
    path.reverse()
    
    return distances[end], path

# Example 1: Simple city map
print("\\nExample 1: Finding shortest path between cities")
print("-" * 50)

# This represents a simple map where:
# - Each city is connected to some other cities
# - The numbers represent distances between cities
simple_graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2},
    'E': {'C': 10, 'D': 2}
}

distance, path = dijkstra_simple(simple_graph, 'A', 'E')
print(f"\\nResult: Shortest distance from A to E: {distance}")
print(f"Path taken: {' -> '.join(path)}")
