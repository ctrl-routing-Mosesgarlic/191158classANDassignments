tup = (1, 2, 3, 4, 5)
print(f"Original tuple: {tup}")
print(f"First element (tup[0]): {tup[0]}")
print(f"Last element (tup[-1]): {tup[-1]}")

mixed_tup = (1, "apple", 3.14)
print(f"Type of mixed_tup: {type(mixed_tup)}")

# Single element tuple should have a trailing comma
single = (5,)
print(f"Type of single-element tuple: {type(single)}")

# Slicing
print(f"Sliced tuple from index 1: {tup[1:]}")

# Concatenation
a = (1, 2)
b = (3, 4)
print(f"Concatenation of a and b: {a + b}")

# Repetition
print(f"Tuple a repeated 3 times: {a * 3}")
