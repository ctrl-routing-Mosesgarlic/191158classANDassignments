class BankAccount:
    def __init__(self, balance):
        self.__balance = balance

    def get_balance(self):
        return self.__balance


    def set_deposit(self, amount):
        self.__balance += amount



newaccount = BankAccount(10000)
print(newaccount.get_balance())