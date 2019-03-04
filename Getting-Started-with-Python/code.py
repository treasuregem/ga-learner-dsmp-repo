# --------------
# Code starts here

# Creating Lists
class_1=['Geoffrey Hinton','Andrew Ng','Sebastian Raschka','Yoshua Bengio']
class_2=['Hilary Mason','Carla Gentry','Corinna Cortes']

# Concatinate
new_class=class_1 + class_2

# Printing New Class
print(f"New Class: {new_class}")

# Add an element in new class
new_class.append('Peter Warden')

# Printing New Class
print(f"New Class: {new_class}")

# Remove an element in new class
new_class.remove('Carla Gentry')

# Printing New Class
print(f"New Class: {new_class}")

# Code ends here


# --------------
# Code starts here

# Creating Dictionary
courses = {'Math':65, 'English':70, 'History':80, 'French':70, 'Science':60}

# See the marks obtained in each subject
for x in courses:
    print(f"{x}: {courses.get(x)}")

# Add marks for total
total=0
for x in courses:
    total=total+courses[x]

# Print total
print(f"Total: {total}")

# Calculate Percentage
max_marks=500
percentage=(total/max_marks)*100

# Print percentage
print(f"Geoffrey Hinton's Percentage: {percentage}%")
# Code ends here


# --------------
# Code starts here

# Creating Dictionary
mathematics={'Geoffrey Hinton':78,'Andrew Ng':95,'Sebastian Raschka':65,'Yoshua Benjio':50,'Hilary Mason':70,'Corinna Cortes':66,'Peter Warden':75}

# Topper in Maths
topper=max(mathematics,key=mathematics.get)

# Print topper student name
print(f"{topper} scored highest marks in Mathematics.")

# Code ends here  


# --------------
# Given string
topper = 'andrew ng'


# Code starts here

# Seperating the first and last name
first_name=topper.split(' ',topper.count(' '))[0]
last_name=topper.split()[1]

# Full name
full_name = last_name + ' ' + first_name

# Convert to uppercase
certificate_name=full_name.upper()

 # Print certificate name
print(certificate_name)


# Code ends here


