def number(n):

    n = int(input("Enter the number n: "))

    # for i in range (n, 0, -1):
    #     print(f"Number n{i}")
    while n > 0:
        print(f"Number n{n}")
        n -= 1

number(8)