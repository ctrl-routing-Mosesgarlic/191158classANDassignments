n_terms = int(input("How many terms the user wants to print"))

#first two terms
n_1 = 0
n_2 = 1

count = 0

#now we will check if the number of the terms is valid or not
if n_terms <= 0:
    print("Please enter a positive integer")