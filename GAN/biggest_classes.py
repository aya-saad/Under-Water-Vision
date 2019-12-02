import os
dir = "2014"

classes = [name for name in os.listdir(dir) if name != ".DS_Store"]
result = {}

for label in classes:
    n = len([name for name in os.listdir("{}/{}".format(dir, label)) if
             os.path.isfile(os.path.join(dir, label, name))])
    if n > 2000:
        result[label] = n

print(result)