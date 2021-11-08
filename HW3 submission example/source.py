import numpy as np

# set seed
rng = np.random.default_rng(seed=42)

a = rng.integers(2, size=(100,50))

text = ""
for i in range(100):
    line = "".join(["1" if b else "0" for b in a[i]])
    text += line
    text += "\n"

# output text is always the same
print(text)
with open("fourmax.txt", "w") as f:
    f.write(text)
