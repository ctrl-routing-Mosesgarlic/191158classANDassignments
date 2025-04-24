# compile time polymorphism or overloading

class Dog:
    def make_sound(self):
        print("Dog sound")


class Cat:
    def make_sound(self):
        print("Cat sound")


def animal_sound(animal):
    animal.make_sound()

cat = Cat()
dog = Dog()

animal_sound(cat)
animal_sound(dog)