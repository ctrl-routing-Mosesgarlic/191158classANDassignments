# 1. Count down numbers from n to 1 using a while loop
# n = int(input("Enter a number to count down from: "))

# while n >= 1:
#     print(n)
#     n -= 1

# 2. Count down from n to 1 using recursion
# def countdown(n):
#     if n >= 1:
#         print(n)
#         countdown(n - 1)

# n = int(input("Enter a number to count down from: "))
# countdown(n)

# 3. Use while loop to count down within a range (user inputs start and end)
# start = int(input("Enter the starting number: "))
# end = int(input("Enter the ending number: "))

# while start >= end:
#     print(start)
#     start -= 1

# # 4. Use recursion to count down within a range


# def countdown_range(start, end):
#     if start >= end:
#         countdown_range(start - 1, end)
#         print(start)
        
# start = int(input("Enter the starting number: "))
# end = int(input("Enter the ending number: "))
# countdown_range(start,end)

# # 5. Recursive function that returns product using repeated addition
# def multiply(a, b):
#     if b == 0:
#         return 0
#     elif b > 0:
#         return a + multiply(a, b - 1)
#     else:  # handle negative b
#         return -multiply(a, -b)
    
# x = int(input("Enter the first number: "))
# y = int(input("Enter the second number: "))

# print(multiply(x, y))	

# def reverse_string(word):
#     # Base case: if the word is empty or just one letter, return it
#     if len(word) <= 1:
#         return word
#     else:
#         # Take the rest of the word (except the first letter), reverse it,
#         # and then add the first letter at the end
#         return reverse_string(word[1:]) + word[0]

# # Try it with a word
# word = "cat"
# reversed_word = reverse_string(word)
# print(reversed_word)

#7. reccursive function that determines whether a given number n 
# is a prime number by checking for divisiblility by intergers less then n
# def is_prime(n, divisor=None):
#     if divisor is None:
#         divisor = n - 1
#     if n <= 1:
#         return False
#     if divisor == 1:
#         return True
#     if n % divisor == 0:
#         return False
#     return is_prime(n, divisor - 1)

# # Test
# num = int(input("Enter a number(n) to check if it's prime: "))
# if is_prime(num):
#     print(num, "is a prime number.")
# else:
#     print(num, "is not a prime number.")
