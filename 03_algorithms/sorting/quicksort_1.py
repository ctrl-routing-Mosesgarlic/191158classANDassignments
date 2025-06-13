def quick_sort(values):
    # If the list has 1 or no elements, it's already sorted
    if len(values) <= 1:
        return values

    # pivot (first element)
    pivot = values[0]

    # Split the rest of the list into two parts
    # left part: numbers less than or equal to pivot
    left = []

    # right part: numbers greater than pivot
    right = []

    for number in values[1:]:  # start from index to skip the pivot
        if number <= pivot:
            left.append(number)
        else:
            right.append(number)

    # Recursively sort both parts and put them together
    return quick_sort(left) + [pivot] + quick_sort(right)

# function call
list_values = [33, 3, 10, 55, 26, 64, 12, 5]
print("Before sorting:", list_values)
sorted_list = quick_sort(list_values)
print("Sorted List:", sorted_list)
