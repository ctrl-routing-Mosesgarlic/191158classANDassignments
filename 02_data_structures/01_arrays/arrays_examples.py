import array

# Create an array of integers
arr = array.array('i', [1, 2, 3, 4])

# Append/add element
arr.append(5)

# or
arr.remove(2)
print(f"  After remove(2): {arr.tolist()}")

# Pop element/ remove 5
arr.pop()

arr.insert(2, 25)
print(f"  After insert(2, 25): {arr.tolist()}")
