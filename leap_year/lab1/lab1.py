

n = int(input("Enter a number"))

factorial = 1

if(n > 1):
    for i in range (1, n+1):
        factorial = factorial * i
        
print("Factorial number is : ", factorial)