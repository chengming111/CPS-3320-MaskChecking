class Person():

    def __init__(self, name = 'Apple', age=18, weight=60, height=160):
        self.name = name
        self.age = age
        self.weight = weight
        self.height = height  
    def input_person_data(self,name, age, weight, height):
        self.name =  name
        self.age = age
        self.weight = weight
        self.height = height
    def get_person_data(self):
        return "name:",self.name,"age:",self.age, "weight:",self.weight,"height:",self.height

def main():
    
    Person_1 = Person()
    Person_2 = Person("Amy",21, 55, 167)

    print("Person 1:", Person_1.get_person_data())
    print("Person 2:",Person_2.get_person_data())
    

if __name__ == "__main__":
    main()
