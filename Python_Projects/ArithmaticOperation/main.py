name = input ('Enter Your Name :')

print ('Hello, ' +name, 'This is a program to perform arithmetic operations.')

age = int(input ('Enter your Age :'))

print ('You will be', age +1, 'Next year')

print ('What operation do you want to perform?')

print ('1.Addition ', add)
print ('2.Subtraction ', sub)
print ('3.Multiplication ', mul)
print ('4.division ', div)

op = add, sub, mul, div

x, y = input ('Enter first value :''\nEnter second value :')

z = x + y
print('The addition is, ', z)

z = x - y
print ('The subtraction is, ', z)

z = x * y
print ('The multiplication is, ', z)

z = x / y
print('The division is, ', z)

print("Enter valid operation!")
