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

    def calculate_day_of_the_week(self):
        h = (self.q + math.floor(13 * (self.m + 1) / 5) + self.K +
             math.floor(self.K / 4) + math.floor(self.J / 4) + 5 * self.J)
        h = int(h % 7)  # Convert to integer for indexing

        # Convert h to day name
        days = ["Saturday", "Sunday", "Monday", "Tuesday",
                "Wednesday", "Thursday", "Friday"]
        return days[h]

    def summary(self):
        day_name = self.calculate_day_of_the_week()
        return (f"Date: {self.year}-{self.month:02d}-{self.day:02d} "
                f"falls on a {day_name}")


def get_user_date():
    while True:
        try:
            year = int(input("Enter the year (e.g., 2023): "))
            month = int(input("Enter the month (1-12): "))
            day = int(input("Enter the day (1-31): "))

            # Basic date validation
            if month < 1 or month > 12:
                print("Month must be between 1 and 12")
                continue
            if day < 1 or day > 31:
                print("Day must be between 1 and 31")
                continue

            return year, month, day
        except ValueError:
            print("Please enter valid numbers for year, month, and day")


if __name__ == "__main__":
    print(">>>> Day of the Week Calculator <<<<")
    print("Enter a date to find out what day of the week it was/is")
    print("=" * 50)

    year, month, day = get_user_date()
    date_calc = DateCalculator(year, month, day)

    print("\n" + "=" * 50)
    print(date_calc.summary())
    print("=" * 50)