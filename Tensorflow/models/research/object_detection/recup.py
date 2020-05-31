#!/usr/bin/python3.7

import os
import sys
import glob

def usage():
    print("usage: python recup.py\nrecoup the last checkpoint name and change the pipeline.config checkpoint field with this name\n")
    exit(1)

path_to_export = "/home/tom/TensorFlow/models/research/object_detection/export/"

if __name__ == "__main__":
    if len(sys.argv) != 1:
        usage()
    checkpoint = sorted(glob.glob(path_to_export + "checkpoint/model.ckpt-*"))
    maxNumber = sorted([fileName.split(".")[1][5:] for fileName in checkpoint])[-1]
    f = open(path_to_export + "pipeline.config", "r")
    pipeline = f.readlines()
    pipeline[157] = "\tfine_tune_checkpoint: \"" + path_to_export + "checkpoint/model.ckpt-" + maxNumber + "\"\n"
    f.close()

    ftmp = open(path_to_export + "pipeline.config.tmp", "w")
    for i in range(len(pipeline)):
        ftmp.write(pipeline[i])
    ftmp.close()

    os.rename(path_to_export + "pipeline.config.tmp", path_to_export + "pipeline.config")

    exit(0)