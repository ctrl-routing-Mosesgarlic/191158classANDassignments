""""
 ─────────────────────────────────────────────────────────────────────────────

    NB: IMPLEMENT THE LOGIC TO PRINT THE DISCOUNTED AND TAXED CART ON TERMINAL, ON LINES 103 & 113

 ─────────────────────────────────────────────────────────────────────────────

As discussed in today's Class, this code will attempt to showcases the concepts of OOP as implemented in Python,
 Notice we have 3 classes in this code, One super class and two children/subcalsess .

[Inheritance: ]  DiscountShoppingCart and TaxedShoppingCart classes inherit
    all methods and properties from the base class ShoppingCart.
    They don’t have to re-implement add_item, remove_item, or summary() methods, they get that for free.

[ Polymorphism:] The 3 classes share the same interface (i.e calculate_total(), summary() are shared).

    This function --> checkout(cart: ShoppingCart), treats all as ShoppingCart instances,
    but each call to calculate_total() dispatches to the correct overridden method at the runtime.

Method Overriding

[Method Overriding:]  The 2 subclasses, i.e DiscountShoppingCart and TaxedShoppingCart, override only the part of the superclass  behavior
     they only change a few modules i.e (applying a discount or tax),
      while delegating the rest which are super().calculate_total().

"""
class ShoppingCart:

    def __init__(self):
        self.items = []

    def add_item(self, item_name: str, qty: int, unit_price: float):

        self.items.append((item_name, qty, unit_price))

    def remove_item(self, item_name: str):

        for item in self.items:
            if item[0] == item_name:
                self.items.remove(item)
                break

    def calculate_total(self) -> float:

        total = 0.0
        for name, qty, price in self.items:
            total += qty * price
        return total

    def cart_contents(self):

        print("Cart Contents: ")
        for name, qty, price in self.items:
            print(f"  {name}:- {qty} @ Ksh {price:.2f} each")
        print(f"Total: Ksh {self.calculate_total():.2f}\n")


# Subclass 1: applies a discount rate
class DicountedCart(ShoppingCart):
    def __init__(self, discount_rate: float):
        super().__init__()

        self.discount_rate = discount_rate  # e.g. 0.10 for 10%

    def calculate_total(self) -> float:
        """Override this method to apply a discount to the initial total."""
        initial_total = super().calculate_total()
        discount = initial_total * self.discount_rate
        return initial_total - discount

# Subclass 2: applies a sales tax, to the parent class
class TaxedCart(ShoppingCart):
    def __init__(self, tax_rate: float):
        super().__init__()
        self.tax_rate = tax_rate

    def calculate_total(self) -> float:
        """Override to add tax on the initial total."""
        initial_total = super().calculate_total()
        tax = initial_total * self.tax_rate
        return initial_total + tax

# A polymorphic function that will work on any class
def checkout(cart: ShoppingCart):
    cart.cart_contents()
    print(f"Total amount: Ksh  {cart.calculate_total():.3f}\n")

# IMPLEMENTATION

if __name__ == "__main__":
    #First we instantiate an Ordinary Cart
    obj_cart = ShoppingCart()
    obj_cart.add_item("Papaya", 76, 6.20)
    obj_cart.add_item("Orange", 96, 11.50)
    obj_cart.add_item("Kiwi", 85, 9.60)
    print(">>> Ordinary Cart Without Tax & Discount <<<")
    checkout(obj_cart)

    # 2) Instantiating and Applying Discount
    """The discount_rate can be any discount of your choice"""
    disc_cart = DicountedCart(discount_rate = 0.15)
    disc_cart.add_item("Papaya", 76, 6.20)
    disc_cart.add_item("Orange", 96, 11.50)
    disc_cart.add_item("Kiwi", 85, 9.60)
    print(">>> Applying a 15% Discount <<<")

    # ─────────────────────────────────────────────────────────────────────────────
    """IMPLEMENT LOGIC TO PRINT THE DISCOUNTED CART ON TERMINAL"""

    # 3) Applying Tax
    taxed_cart = TaxedCart(tax_rate=0.07)
    taxed_cart.add_item("Papaya", 5, 2.00)
    taxed_cart.add_item("Orange", 96, 11.50)
    taxed_cart.add_item("Kiwi", 3, 1.50)
    print(">>> Applying a 12% Tax  <<<")
    checkout(taxed_cart)
    # ─────────────────────────────────────────────────────────────────────────────
    """IMPLEMENT LOGIC TO PRINT THE TAXED CART ON TERMINAL"""