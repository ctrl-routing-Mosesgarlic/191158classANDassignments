a = {1, 2, 3}
b = {3, 4, 5}

print(f"  a: {a}")
print(f"  b: {b}")

# {1, 2, 3, 4, 5}
print(f"  Union (a | b): {a | b}")

# {3}
print(f"  Intersection (a & b): {a & b}")

# {1, 2}
print(f"  Difference (a - b): {a - b}")

# {1, 2, 4, 5}
print(f"  Symmetric Difference (a ^ b): {a ^ b}")

# True
print(f"  2 in a: {2 in a}")

# False
print(f"  5 in a: {5 in a}")

print(f"  Length of a: {len(a)}")

for item in a:
    print(f"  Item: {item}")

nums = [1, 2, 2, 3, 4, 4]
unique_nums = set(nums)
print(f"  Unique elements: {unique_nums}")
