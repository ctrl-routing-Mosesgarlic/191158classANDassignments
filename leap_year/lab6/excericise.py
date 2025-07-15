# #  1st question
# n= int(input("ENTER NUMBER"))
# while n>=1:
#     print(n)
#     n -=1

# #Second question

# def countdown(n):
#     if n==1:
#         print(1)

#     else:
#         print(n)
#         countdown(n-1)

# countdown(10)

# #Third question
# m = int (input("Enter lower range"))
# k = int (input("Enter upper range"))

# while m<=k:
#     print(k)
#     k-=1

#Fourth Question
def rangecount(j,h):
    if j==h:
        print(j)
    
    else:
        print(h)
        rangecount(j,h-1)

rangecount(5,10)

# #Fifth question
# def Product(e,f):
#     if f==1:
#         return e
    
#     else:
#         return e + Product(e,f-1)
    
# print(Product(3,4))



def reverse_string(word):
    # Base case: if the word is empty or just one letter, return it
    if len(word) <= 1:
        return word
    else:
        # Take the rest of the word (except the first letter), reverse it,
        # and then add the first letter at the end
        return reverse_string(word[1:]) + word[0]

# Try it with a word
word = "cat"
reversed_word = reverse_string(word)
print(reversed_word)


def is_prime(n, divisor=None):
    if divisor is None:
        divisor = n - 1
    if n <= 1:
        return False
    if divisor == 1:
        return True
    if n % divisor == 0:
        return False
    return is_prime(n, divisor - 1)

# Test
num = int(input("Enter a number(n) to check if it's prime: "))
if is_prime(num):
    print(num, "is a prime number.")
else:
    print(num, "is not a prime number.")


