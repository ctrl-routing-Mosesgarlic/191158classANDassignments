def is_leap(year: int) -> bool:
    """Do the check."""
    if (year % 4) == 0:
        # If divisible by 4 and also by 100 do these tests
        if (year % 100) == 0:
            # If it is divisible by 4 and 400 it is leap year
            if (year % 400) == 0:
                leap_year = True
            else:
                leap_year = False
        else:
            leap_year = True
    else:
        leap_year = False

    return leap_year