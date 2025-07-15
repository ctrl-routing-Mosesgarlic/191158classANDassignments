# int 
# bool 
# list 
# tuple 
# float 
# str 
# dict 
#DUCK TYPING

#primitive data structures are like atoms 

"""this 
is 
a 
multiline 
comment
"""


num = 10
name = "John Doe"
avg = 98.9
names = ["Mary" , "John"]

num: int = 10
name: str = "John Doe"
avg: float = 98.9
names: list = ["Mary" , "John"] #it is immutable
cars: tuple = ("volvo" , "benz")       #it is mutable
marks: set = [1, 2 , 3]
is_passed = True
is_adult = False 


"""print(num)
print(name)
print(avg)
print(names)

print(f'Name {name} Average {avg}')"""

"""for i in range(10):
    for j in range(i):
        print(j)"""

# for i in names:
#     print(i)

# names.append("Ruth")
# print(names)

# names.pop(1)
# print(names)

# # names.clear()
# print("After clearing" , names)

# x: int = 5

# while x > 2:
#     print(name)
#     break

"""def func() ->None:
    print(90)
    print("Average")
    age: int = 20
    
    for j in "names":
        print(j)
        if age >= 20:
            print("Adult")
    
func()"""

# std: dict = {"name": "Alice" , "Age": 21}
# print(std)


students: list = [
    {"student_name": "Alice" , "marks": [70,89,67]},
    {"student_name": "Bob" , "marks": [67,90,56]},
    {"student_name": "Bob" , "marks": [89,91,85]},
    {"John": "Charlie" , "marks": [89,91,85]},
]

# students_two: list = [
#     {"name": "Alice" , "marks": [99,89,67]},
#     {"Name": "Bob" , "marks": [90,90,56]},
#     {"Name": "Bob" , "marks": [94,91,85]},
# ]
def average(marks: list[int]) -> float:
    return sum(marks / length(marks))

    
