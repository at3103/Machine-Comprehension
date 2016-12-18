import operator
import math

word_vectors_filename = "../data/glove/glove.6B.50d.txt"

new_filename = "../data/glove/glove.6B.50d_normalized.txt"

def vector_magnitude(vec):
	magnitude = 0
	magnitude = sum(map(operator.mul,vec,vec))
	magnitude = math.sqrt(magnitude)
	return magnitude

# Read in word vectors
with open(word_vectors_filename, 'r') as f, open(new_filename, 'w') as n:
    for line in f:
        line_list = line.split()
        word = line_list.pop(0)
        vector = line_list
        vector = map(float, vector)

        # Normalize value
        magnitude = vector_magnitude(vector)
        norm_value = [val / magnitude for val in vector]
        norm_value = map(str, norm_value)

        # Write to the other file
        n.write("%s %s\n" % (word, " ".join(norm_value)))
