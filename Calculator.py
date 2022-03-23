class Calculator:
    def __init__(self):
        self.num1 = int(input("Enter the first number :- "))
        self.num2 = int(input("Enter the second number:- "))

    def addition(self):
        r = self.num1 + self.num2
        result = 'The sum of the '+str(self.num1) + 'and' + str(self.num2) + 'is' + str(r)
        print(result)
        return result

    def subtract(self):
        r = self.num1 - self.num2
        result = 'The difference of the '+ str(self.num1) +' and '+str(self.num2)+' is '+str(r)
        print(result)
        return result

    def multiply(self):
        r = self.num1 * self.num2
        result = 'The difference of the ' + str(self.num1) + 'and' + str(self.num2) + 'is' + str(r)
        print(result)
        return result

    def divide(self):
        r = self.num1 / self.num2
        result = 'The division of the ' + str(self.num1) + 'and' + str(self.num2) + 'is' + str(r)
        print(result)
        return result


if __name__ == '__main__':
    operation = Calculator()
    operation.addition()
    operation.multiply()
    operation.subtract()
    operation.divide()
