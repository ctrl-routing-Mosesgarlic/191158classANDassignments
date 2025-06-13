import random

def binary_search_iterative(inputs, target):
    low = 0
    high = len(inputs) - 1

    while low <= high:
        mid = (low + high) // 2

        if inputs[mid] == target:
            return mid
        elif inputs[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1

# Generate 5 random integers between 1 and 100
values = sorted([random.randint(1, 100) for _ in range(10)])
print("Ordered:", values)

# get value to search
target = int(input("Enter number to search : "))
result = binary_search_iterative(values, target)

# result
if result != -1:
    print(f"Value {values[result]} found at index {result}")
else:
    print(f"Value {target} not found in the list")

