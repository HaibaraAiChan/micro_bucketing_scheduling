import remove_duplicates

my_list = [1, 2, 2, 3, 3, 4, 4, 5, 5]
new_list = remove_duplicates.remove_duplicates(my_list)

print(new_list)  # Outputs: [1, 2, 3, 4, 5]

# this is not efficient than dict
