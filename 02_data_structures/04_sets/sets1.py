my_set = {1, 2, 3, 4, 5}

#use set(), not {}
empty_set = set()
print(type(my_set))

print(f"  my_set: {my_set}")


# insert/ add element - cant have duplicates hence second 5 not displayed
my_set.add(5)
print(f"  After adding 5: {my_set}")

# Error if element not found
# my_set.remove(112)
print(f"  After removing 112: {my_set}")

# No error if element not found
my_set.discard(112)
print(f"  After discarding 112: {my_set}")
