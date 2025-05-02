import heapq

pq = []
heapq.heappush(pq, (2, "Bob"))
heapq.heappush(pq, (1, "Alice"))
heapq.heappush(pq, (3, "Charlie"))

print(f"  Priority queue: {pq}")

priority_item = heapq.heappop(pq)
print(f"  Served: {priority_item}")

#  Peeking Without Removing
if pq:
    print(f"  Peek: {pq[0]}")

# Push and Pop Simultaneously
item = heapq.heappushpop(pq, (0, "Davis"))  # Push and immediately pop the smallest
print(f"  heappushpop result: {item}")
print(f"  Queue now: {pq}")

# # Pops the smallest and pushes new
print(f"  After heapreplace: {pq}")
heapq.heapreplace(pq, (3, "Evans"))
print(f"  After heapreplace: {pq}")
