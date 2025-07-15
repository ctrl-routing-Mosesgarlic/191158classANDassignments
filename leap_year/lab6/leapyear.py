def is_leap(year: int) -> bool:  # Define a function that takes 'year' as input and returns True/False
    if (year % 4) == 0:  # Check if the year is divisible by 4
        if (year % 100) == 0:  # If divisible by 4, check if also divisible by 100
            if (year % 400) == 0:  # If divisible by 100, check if also divisible by 400
                leap_year = True  # If divisible by 400, it's a leap year
            else:
                leap_year = False  # If divisible by 100 but not 400, not a leap year
        else:
            leap_year = True  # If divisible by 4 but NOT by 100, it's a leap year
    else:
        leap_year = False  # If not divisible by 4, it's not a leap year

    return leap_year  # Return the result (True or False)


# Example usage
year = int(input("Enter a year: "))  # Ask the user to enter a year and convert it to an integer
if is_leap(year):  # Call the function and check if it returns True
    print(f"{year} is a leap year! 🎉")  # If True, print it's a leap year
else:
    print(f"{year} is not a leap year.")  # If False, print it's not a leap year
