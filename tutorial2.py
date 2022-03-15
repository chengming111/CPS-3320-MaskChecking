print("Enter an integer")
num = int(input())

if num <= 0:
    num = int(input)
else:
    if num % 2 == 0:
        print("number",num,"is multiple of 2,4")
    elif num % 3 == 0:
        print("number",num,"is multiple of 3")
    elif num % 4 == 0:
        print("number",num,"is multiple of 4")
    else:
        print("number",num,"is not a multiple of 2,3,4")

print("End of Program")
