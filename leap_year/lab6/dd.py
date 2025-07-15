def factorial(n):
    if n==0 or n==1:
        return 1
    else:
        return n*factorial(n-1)
    
 
print(factorial(4))

def findPower(number,exponent):
    resultPower = 1
    for _ in range(1,exponent+1):
      resultPower=resultPower*number
    return resultPower

print(findPower(2,3))

def power(base,exp):
    if(exp==1):
        return(base)
    if(exp!=1):
        return(base*power(base,exp-1))
    

base=int(input("Enter base: "))
exp=int(input("Enter exponential value: "))
print("Result",power(base,exp))


def fibonacci_recursive(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# Print the first 10 Fibonacci numbers
n_terms = 10
series = [fibonacci_recursive(i) for i in range(n_terms)]
print(series)








