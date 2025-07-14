class ShoppingCart:

    def __init__(self):
        pass
    def area(self, radius) -> float:
        return 3.14 * (radius * radius)
    
obj1: ShoppingCart = ShoppingCart()
obj2: ShoppingCart = ShoppingCart()

print(obj1.area(7))
print(obj2.area(7.7))