def merge_sort(values):
    if len(values) <= 1:
        return values

    # Divide the list into halves
    mid = len(values) // 2
    left_half = merge_sort(values[:mid])
    right_half = merge_sort(values[mid:])

    # Merge the sorted halves
    return merge(left_half, right_half)

def merge(left, right):
    merged = []
    i = j = 0

    # Compare elements and merge them in sorted order
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # Add any remaining elements
    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged

# call the function
list_values = [38, 27, 43, 3, 9, 82, 10]
sorted_list = merge_sort(list_values)
print("Sorted array:", sorted_list)
