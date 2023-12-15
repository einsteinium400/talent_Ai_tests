import numpy as np
from collections import Counter
import ast
def create_new_list(cluster):
    # Extract lists from the 6th index of each vector
    lists_at_sixth_index = [ast.literal_eval(vector[5]) for vector in cluster]

    # Flatten the lists and count occurrences of each value
    flattened_list = [item for sublist in lists_at_sixth_index for item in sublist]
    value_counts = Counter(flattened_list)

    # Find values that appear in at least 50% of the lists
    threshold = len(cluster) / 2
    new_list = [value for value, count in value_counts.items() if count >= threshold]

    return new_list

# Example usage
cluster = np.array([
    [1, 2, 3, 4, 5, "['f', 'b']"],
    [2, 3, 4, 5, 6, "['b', 'c']"],
    [3, 4, 5, 6, 7, "['a', 'c']"],
    [4, 5, 6, 7, 8, "['a', 'b']"],
    [5, 6, 7, 8, 9, "['b', 'c']"]
])

result = create_new_list(cluster)
print(result)
