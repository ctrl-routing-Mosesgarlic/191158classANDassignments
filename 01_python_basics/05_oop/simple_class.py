class Dog:
    def __init__(self, name):
        self.new_name = name

    def bark(self):
        print(f" The dogs name is {self.new_name}")


mydog = Dog("ABC")
mydog.bark()



"""
magic methods

__add__

__str__

__len__

__eq__
"""