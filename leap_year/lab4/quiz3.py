def count_down(n, m):

    #where n is the end number is n and the start the number is m
    n = int(input("Enter the number n: "))

    m = int(input("Enter the number m: "))

    for i in range (n, m, -1):
        while n > m:
            print(f"Number n{n}")
            n -= 1

count_down(8, 1)