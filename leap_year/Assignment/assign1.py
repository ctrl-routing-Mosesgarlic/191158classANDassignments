import math

class DateCalculator:

    def __init__(self, q, m ,y, k, j):
        self.q = q
        self.m = m
        self.y = y
        self.k = k
        self.j = j

    """
    Let:

- `q` = day of the month  
- `m` = month (3 = March, 4 = April, ..., 12 = December; January and February are counted as 13 and 14 of the previous year)  
- `Y` = year (adjusted if month is Jan or Feb)  
- `K` = year of the century (`year % 100`)  
- `J` = zero-based century (`year // 100`)

Then the formula is:

## h = (q + ⌊13(m + 1) / 5⌋ + K + ⌊K / 4⌋ + ⌊J / 4⌋ + 5J) % 7
"""
    def calculate_day_ofthe_week(self):

        h = (q + math.floor(13 * (m + 1) / 5) + K + math.floor(K / 4) + math.floor(J / 4) + 5 * J) % 7


        return h

    def summary(self):
        print("cart content")

        print(f"Date: TIMESTAMP{self.calculate_day_ofthe_week():.3f}")

if __name__ =="__main__":

    Date: DateCalculator = DateCalculator()

    Date.calculate_day_ofthe_week()

    print(">>>>my Cart<<<<")
    print(Date.summary())
