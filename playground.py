#playground.py
import numpy as np
import json
import math

k = 20
for val in [.0345, .167]:
	print(math.ceil(val * k))


# oppol = None

# with open('policy.txt', "r") as f:
# 	oppol = json.load(f)

# count = 0
# total = 0
# for k,v in oppol.items():
# 	total += 1
# 	if v != 1:
# 		count += 1
# print(count)
# print(total)
# print(count / total)