from math import e

def sigmoid(z):
    return 1/(1+e**-z)

print(sigmoid(.00001))
print(sigmoid(10000))
print(sigmoid(-1))
print(sigmoid(-10))
print(sigmoid(-2.0))