import math


class DateCalculator:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

        # Adjust for January and February
        if month < 3:
            self.m = month + 12
            self.y = year - 1
        else:
            self.m = month
            self.y = year

        self.q = day
        self.K = self.y % 100  # year of the century
        self.J = self.y // 100  # zero-based century

    @property
    def calculate_day_of_the_week(self):
        h = (self.q + math.floor(13 * (self.m + 1) / 5) + self.K + math.floor(self.K / 4) + math.floor(
            self.J / 4) + 5 * self.J)
            h = h % 7

        # Convert h to day name
        days =["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        return days[int(h)]

    def summary(self):
        day_name = self.calculate_day_of_the_week
        return f"Date: {self.year}-{self.month:02d}-{self.day:02d} falls on a {day_name}"


if __name__ == "__main__":
    # Example: September 15, 1589
    date_calc = DateCalculator(year=1589, month=9, day=15)

    print(">>>> Day of the Week Calculator <<<<")
    print(date_calc.summary())