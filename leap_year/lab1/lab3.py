"""
Creating a function that returns the power of a number.
It accepts Number, and exponent values as parameters
"""
def findPower(number , exponent):
    #initializing a variable with 1(it stores the resultant power)
    resultPower = 1
    
    #traversing in the range from 1 to a given exponent+1
    for i in range(1, exponent+1):
    
        #multiplying the result with the given number
        resultPower = resultPower*number
    
    #returning the resultant power
    
    return resultPower

print(findPower(2,3))
