import pickle

# Open file
input = open('./data/glove.6B.300d.txt')

# Create dict of edge type -> index
embeddings = {}

# While there are more lines, read lines
for line in input:
    # Split on tab
    split_line = line.split(' ')
    embeddings[split_line[0]] = split_line[1:]

input.close()

# Write dict to file
output = open('./data/glove_300.pickle', 'wb')
pickle.dump(embeddings, output)
output.close()

