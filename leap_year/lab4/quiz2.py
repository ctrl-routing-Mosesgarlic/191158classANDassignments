def count_down(n):
    print(f"Number n: {n} ")

    if n == 1:
        return
    else:
        count_down(n-1)

count_down(10)