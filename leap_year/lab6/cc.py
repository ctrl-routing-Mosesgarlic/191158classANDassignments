number: int = 10
name: str = "Jane doe" 
age: float = 10.1
names: list = ["Mary","Jane"] 

# print ("My name is",name,"I am ",age,"Years old")
# print(age) #dummy parameter/argument
# print(10) #actual parameter
# print(f'name {name} age {age}')

# is_passed = True
# is_passed = False

# for i in range(10):
#     print(i)

#     for i in names:
#         print(i)


# names.append("Ruth")
# print(names)

# names.pop(1)
# print(names)Student_

# names.clear()
# print(names)

# x:int = 0

# while x < 2:
#     print(names)
#     print(names)
#     break
# print(names)

# def func()->None:
#     print(10)
#     age = 20

#     print("Average")

#     for j in "names":
#         print(j)
#         if age>=20:
#             print("Adult")

#     func()

std:dict = {"name":"alice", "Age":20}
#print(std)

students: list = [
    {"Student_Name":"Alice","Marks":[10,20,230]},
    {"Student_Name":"Bob","Marks":[30,55,78]},
    {"Student_Name":"Kia","Marks":[90,85,44]},
]

students: list = [
    {"Student_Name":"Alice","Marks":[10,20,230]},
    {"Student_Name":"Bob","Marks":[30,55,78]},
    {"Student_Name":"Kia","Marks":[90,85,44]},
]

def average(Marks:list[int])->float:
    return sum(Marks) /len(Marks)

for student in students:
    name = student["Student_Name"]
    marks = student["Marks"]
    avg = average(marks)
    print(f"{name}'s average marks: {avg:.2f}")



    