student = {
    "name": "Jane",
    "age": 20,
    "grade": "A"
}

print(f"  student: {student}")

# Accessing Values
print(f"  Name: {student['name']}")           # Alice
print(f"  Age: {student.get('age')}")         # 20
print(f"  GPA: {student.get('gpa', 'N/A')}")  # Key not found, returns 'N/A'

# Updating and Adding Values
student["age"] = 21
student["major"] = "Computer Science"

print(f"  Updated student: {student}")

# Removing Key-Value Pairs
del student["grade"]
print(f"  After deletion: {student}")

# Using pop()
major = student.pop("major")
print(f"  Removed major: {major}")
print(f"  After pop: {student}")
