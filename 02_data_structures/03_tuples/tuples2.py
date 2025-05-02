
tup = (1, 2, 2, 3, 4, 5)
print(f"Check type {type(tup)}")

# Looping Through a Tuple
for item in tup:
    print(f"Item: {item}")


print(f"Count of 2 in tup: {tup.count(2)}")

# first index of 3
print(f"Index of 3 in tup: {tup.index(3)}")

# unpack tuples
person = ("Alice", 25)
name, age = person
print(name)
print(age)

# nested tuple
nested = ((1, 2), (3, 4))
print(nested[1][0])
