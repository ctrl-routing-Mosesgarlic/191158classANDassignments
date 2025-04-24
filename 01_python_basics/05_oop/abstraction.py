# abstraction
# run time polymorphism or overriding

from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

    def something_else(self):
        print("Something else")


class Dog(Animal):

    def make_sound(self):
        pass

    def roam(self):
        print("Roam")


zzz = Animal()

# qwe = Dog()
# qwe.make_sound()