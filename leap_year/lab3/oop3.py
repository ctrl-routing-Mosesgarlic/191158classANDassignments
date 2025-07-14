class ShoppingCart:
    
    def __init__(self):
        self.items[]
    

    def add_items(self, item_name: str, qty: str,unit_price: float):
        self.item_name = item_name
        self.qty = qty
        self.unit_price = unit_price
        
        self.items.append((item_name,qty,unit_price))
        
    def remove_item(self, item_name: str, unit_price):
        for item in self.items:
            if items[0] == item_name:
                self.items.remove(items)
            

    def calculate_total (self)-> float:
        total =0
        for name,qty,unit_price in self.items:
            total+= qty*unit_price
        return total
    
    def summary(self):
        print("cart content")
        
        for name,qty,unit_price in self.items:
            print(f"{name}:{qty}@ Ksh{price:.3f}")
            
        print(f"Total:Ksh{self.calculate_total():.3f}")
        
if __name__ =="__main__":
    cart: shoppingCart = shoppingCart()
    
    cart.add_items(item_name: "book"qty: " 45")
    cart.add_items(item_name: "pencils"qty: " 65")
    cart1.add_items(item_name: "rubber"qty: " 45")
    
    print(">>>>my Cart<<<<")
    print(cart.summary())

"""obj: ShoppingCart = ShoppingCart(item_name: "Papaya", qty: 9)
obj1: ShoppingCart = ShoppingCart(item_name: "Guava", qty: 17)

print(obj.item_name)"""
