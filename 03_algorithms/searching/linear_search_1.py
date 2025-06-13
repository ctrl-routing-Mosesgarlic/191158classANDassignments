
def get_values(items, key):
    for item in items:
        if item == key:
            print(f" {item} found")
    return -1

random_values = [5, 1, 44, 7, 90, 12, ]
print(random_values)
key = int(input("Enter number to search : "))
result = get_values(random_values, key)


