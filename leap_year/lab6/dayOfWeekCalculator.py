class DateCalculator: #Define a class `DateCalculator`  
    def __init__(self,year: int, month: int, day: int): #nitialize with `year`, `month`, `day
        self.year = year
        self.month = month
        self.day = day

    def calculate_day_of_week(self) -> str:
        # Adjust month/year for January and February
        if self.month == 1 or self.month == 2:
            self.month += 12
            self.year -= 1

        q = self.day  # Day of the month
        m = self.month  # Month
        Y = self.year  # Year after adjustment

        K = Y % 100  # Year of the century
        J = Y // 100  # Zero-based century

        h = (q + (13 * (m + 1)) // 5 + K + (K // 4) + (J // 4) + 5 * J) % 7 #FORMULA

        # Mapping the result to the day of the week
        days = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        return days[h]  # h gives index for the correct day
    

if __name__ == "__main__":
    year = int(input("Enter year: "))
    month = int(input("Enter month: "))
    day = int(input("Enter day: "))

    calculator = DateCalculator(year, month, day)  # Create an object
    weekday = calculator.calculate_day_of_week()   # Calculate the day

    print(f"The day of the week for {day}/{month}/{year} is {weekday}.")


        