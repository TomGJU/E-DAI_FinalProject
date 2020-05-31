#!/usr/bin/python3.7

### IMPORTANT ###
# If you want to use this script, don't forget to change the differents path to the folders where your data are.

destination_path = "/home/tom/Scripts&Data/DataE&DAI/LabelsTxt"

alone_i = 0
coffee_i = 0

while (alone_i < 34):
    name = "alone-frame" + str(alone_i) + ".txt"
    if (alone_i > 0 and alone_i < 8):
        f = open(name, "x")
    if (alone_i > 24):
        f = open(name, "x")
    alone_i += 1

while (coffee_i < 35):
    name = "coffee-frame" + str(coffee_i) + ".txt"
    if (coffee_i > 0 and coffee_i < 7):
        f = open(name, "x")
    if (coffee_i > 28):
        f = open(name, "x")
    coffee_i += 1
