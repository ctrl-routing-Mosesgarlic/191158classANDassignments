def get_positive_number():
    while True:
        try:
            n = int(input("Enter the end number n: "))
            m = int(input("Enter the start number m: "))

            if n> 0 and n > m:
                return n
            print("Please input valid numbers 'end and start numbers'")
        except ValueError:
            print("Invalid Input")

def countdown_recursive(n, m):
    print(n)
    if n == m:#base case
        return
    else:
        countdown_recursive(n-1, m)

countdown_recursive(20,10)