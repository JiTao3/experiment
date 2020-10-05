import numpy as np

# data = []
with open("data/assertion_dmv.txt", "r") as f:
    lines = f.readlines()
    lines = [[float(j) for j in i.replace('\n','').split(',')]for i in lines]
    
ass_data = np.array(lines, dtype=np.float)

print(np.max(ass_data, axis=-2))
print(np.min(ass_data, axis=-2))
