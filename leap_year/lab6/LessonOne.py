# int
# bool
# list
# tuple
# float
# str
# dict

# DUCK TYPING

                                                                                                                          z
num: int = 10
name: str = "John Doe"
avg:int  = 19.95
names:list = ["carl", "mary"]
cars:tuple =["volvo","CX-5"]
marks:set = [1,2,3]

# print(f'My name is {name} and I have an estimated networth of {num} billion dollars')
# print(avg)

# for i in names:
#     print(i)

#                            LISTS
# names.append("ruth")
# print(names)
#
# names.pop(1)
# print(names)
#
# names.clear()
# print("After clearing" , names)
#
# x:int = 0

# while x<2:
#     print(names)
#     print(names)
#     print(names)
#     print(names)
#     break
#     print(names)70,


# def example() ->None:
#     print(90)
#     age = 20
#
#     print("Average")
#     for j in "names":
#         print(j)
#
#         if age >= 20:
#             print(age)
std: dict = {"Name": "Alice", "Age":21}
# print(std)
students: list = {
    {"Name":"Alice", "Marks": [70,69,68]},
    {"Name":"Makr", "Marks": [71,63,64]},
    {"Name":"John", "Marks": [79,86,45]}

}


def average(marks:list(int))->float:
return sum(marks)/len()
