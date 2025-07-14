import math


class DateCalculator:
    def __init__(self, year, month, day):
        self.day = day
        self.month = month
        self.year = year

        # Adjust month and year for January and February
        if month == 1 or month == 2:
            self.m = month + 12
            self.y = year - 1
        else:
            self.m = month
            self.y = year

        # Calculate K and J components
        self.k = self.y % 100  # Year of the century
        self.j = self.y // 100  # Zero-based century

    """
    Zeller's Congruence Formula:
    Let:
    - `q` = day of the month
    - `m` = month (3 = March, 4 = April, ..., 12 = December; January and February are counted as 13 and 14 of the previous year)
    - `Y` = year (adjusted if month is Jan or Feb)
    - `K` = year of the century (`year % 100`)
    - `J` = zero-based century (`year // 100`)

    Formula: h = (q + ⌊13(m + 1) / 5⌋ + K + ⌊K / 4⌋ + ⌊J / 4⌋ + 5J) % 7
    """

    def calculate_day_ofthe_week(self):
        # Apply Zeller's formula
        h = (self.day +
             math.floor(13 * (self.m + 1) / 5) +
             self.k +
             math.floor(self.k / 4) +
             math.floor(self.j / 4) +
             5 * self.j) % 7

        # Map result to day of week (0 = Saturday, 1 = Sunday, ..., 6 = Friday)
        days = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        return days[h]

    def summary(self):
        day_of_week = self.calculate_day_ofthe_week()
        print("Date Summary:")
        print(f"Date: {self.day}/{self.month}/{self.year}")
        print(f"Day of the week: {day_of_week}")
        return day_of_week


if __name__ == "__main__":
    # Example: Calculate the day of the week for September 15, 1589
    date_calc = DateCalculator(1589, 9, 15)

    print(">>>>Day of the Week Calculator<<<<")
    day = date_calc.calculate_day_ofthe_week()
    print(f"September 15, 1589 was a {day}")

    # You can also try with today's date or any other date
    # For example, to calculate for May 2, 2025:
    today = DateCalculator(2025, 5, 2)
    print("\n>>>>Today's Date<<<<")
    today.summary()