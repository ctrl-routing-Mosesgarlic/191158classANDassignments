import random

def get_values(items, key):
    for index, item in enumerate(items):
        if item == key:
            # if value found return its index
            return index
    # if value not found return -1
    return -1

# Generate 10 unique random numbers from 1 to 100
random_values = random.sample(range(1, 101), 10)
print("Random values:", random_values)

# Get value to search from the user
key = int(input("Enter number to search: "))
result = get_values(random_values, key)

# result
if result != -1:
    print(f"Value {random_values[result]} found at index {result}")
else:
    print(f"Value {key} not found in the list")
