#!/usr/bin/python3.6

### IMPORTANT ###
# If you want to use this script, don't forget to change the differents path to the folders where your data are.

import re
import glob

labelsPath = "/home/tom/Scripts&Data/DataE&DAI/LabelsTxt"

def getData(fo):
    print("===========")
    f = open(fo, "r")
    f1 = f.readlines()
    data = f1[19:-3]
    strin = ""
    for i in data:
        if i[5] == 'm' and i[-5] == 'm':
            strin += i
    print(strin)
    strin = strin.replace("<xmin>", "1 ").replace("\t", "").replace("\n", "").replace("</xmin><ymin>", " ").replace("</ymin><xmax>", " ").replace("</xmax><ymax>", " ").replace("</ymax>1", "\n1").replace("</ymax>", "")
    print(strin)
    return strin
    """print(data)
    for i in range(len(data)):
        data[i] = re.sub("[^0-9]", "", data[i])
    print(data)
    return data"""
    """for x in f1:
        print(x)"""

labels = sorted(glob.glob(labelsPath + "/*.xml"))
print(labels)

getData(labels[0])

for fo in labels:
    data = getData(fo)
    f = open(fo[:-3]+"txt", "w+")
    for i in range(len(data)):
        f.write(data[i])


