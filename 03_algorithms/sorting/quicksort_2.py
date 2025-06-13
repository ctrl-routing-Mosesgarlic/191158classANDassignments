def quick_sort(values):
    if len(values) <= 1:
        return values

    # first element is the pivot
    pivot = values[0]

    left = [x for x in values[1:] if x <= pivot]
    right = [x for x in values[1:] if x > pivot]

    return quick_sort(left) + [pivot] + quick_sort(right)

list_values = [33, 10, 55, 26, 64, 12, 5]
sorted_list = quick_sort(list_values)
print("Sorted array:", sorted_list)
