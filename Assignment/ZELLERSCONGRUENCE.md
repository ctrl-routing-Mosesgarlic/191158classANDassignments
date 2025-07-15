# Day of the Week Calculator – Zeller's Congruence

##  *Which Day of the Week?*

An enduring puzzle in date and calendar calculations is figuring out what day of the week a given date falls on. For example:  
**What day of the week was September 15, 1589?**


---



This task is complex because of:

- Non-uniform month lengths (28–31 days)
- Leap years every 4 years (with exceptions for years divisible by 100 and not 400)
- Calendar reforms (like the switch from Julian to Gregorian)



---

## The Zeller’s Congruence Formula

Zeller’s formula calculates the day of the week for any date in the Gregorian calendar.

Let:

- `q` = day of the month  
- `m` = month (3 = March, 4 = April, ..., 12 = December; January and February are counted as 13 and 14 of the previous year)  
- `Y` = year (adjusted if month is Jan or Feb)  
- `K` = year of the century (`year % 100`)  
- `J` = zero-based century (`year // 100`)

Then the formula is:

## h = (q + ⌊13(m + 1) / 5⌋ + K + ⌊K / 4⌋ + ⌊J / 4⌋ + 5J) % 7

## Task: Solve This Using OOP

- Define a class `DateCalculator`  
- Initialize with `year`, `month`, `day`  
- Adjust month/year for January and February  
- Calculate components: `K` and `J`  
- Use the formula to compute the weekday  
- Return or print the result

---