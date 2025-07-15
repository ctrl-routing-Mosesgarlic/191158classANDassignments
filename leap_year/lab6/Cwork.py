# - Compute marks for every student and display in the terminal
# Sample data: list of students and their marks
# students = [
#     {"name": "Alice", "math": 78, "english": 85, "science": 90},
#     {"name": "Bob", "math": 67, "english": 74, "science": 80},
#     {"name": "Charlie", "math": 92, "english": 88, "science": 95}
# ]
#
# # Compute and display marks
# for student in students:
#     total = student["math"] + student["english"] + student["science"]
#     average = total / 3
#
#     print(f"Student: {student['name']}")
#     print(f"  Math: {student['math']}")
#     print(f"  English: {student['english']}")
#     print(f"  Science: {student['science']}")
#     print(f"  Total Marks: {total}")
#     print(f"  Average: {average:.2f}")
#     print("-" * 30)

#- Loop through the list of dictionaries of students and output John when found
# students = [
#     {"name": "Alice", "age": 18},
#     {"name": "John", "age": 19},
#     {"name": "Charlie", "age": 20}
# ]
#
# # Loop through and find John
# for student in students:
#     if student["name"] == "John":
#         print("John found!")
#         break  # optional: stops after finding John


year_input = input("Enter a year (max 4 digits): ")

# Check if input is all digits and not more than 4 characters
if not year_input.isdigit() or len(year_input) > 4:
    print("Invalid input. Please enter a number with at most 4 digits.")
else:
    year = int(year_input)
    # Leap year logic
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        print(year, "is a leap year.")
    else:
        print(year, "is not a leap year.")
