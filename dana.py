# Your list of lists
original_list = [
    [1, 2, 3, 4, 5, 6, [7, 8, 9, 10]],
    [2, 3, 4, 5, 6, 7, [8, 9]],
    [3, 4, 5, 6, 7, 8, [9, 10]],
]

# Process the 7th index in each sublist and extract them as standalone lists
processed_list = [sublist[6][:4] + [1] * (4 - len(sublist[6])) if len(sublist[6]) < 4 else sublist[6][:4] for sublist in original_list]
averages = [sum(item[i] for item in processed_list) / len(processed_list) for i in range(len(processed_list[0]))]

print("Processed List:", averages)
