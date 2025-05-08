numbers = [3, 1, 4, 1, 5]
print(f"  Original list: {numbers}")

print(f"  Length: {len(numbers)}")
print(f"  Sum: {sum(numbers)}")
print(f"  Min: {min(numbers)}")
print(f"  Max: {max(numbers)}")

numbers.sort()
print(f"  Sorted list: {numbers}")

numbers.reverse()
print(f"  Reversed list: {numbers}")

# List Comprehension
squares = [x**2 for x in range(6)]
print(f"  Squares using list comprehension: {squares}")
