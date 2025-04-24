from abc import ABC, abstractmethod
class Animal:
    @abstractmethod
    def make_sound(self):
        print("Make sound")


class Dog(Animal):
    pass


qwe = Dog()
qwe.make_sound()
