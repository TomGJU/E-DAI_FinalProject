#!/usr/bin/python3.7

### IMPORTANT ###
# If you want to use this script, don't forget to change the differents path to the folders where your data are.

import os
import cv2
import sys
import glob
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from shutil import copyfile
import matplotlib.pyplot as plt
from object_detection.utils import dataset_util

imagesPath = "/home/tom/Scripts&Data/DataE&DAI/Images"
labelsPath = "/home/tom/Scripts&Data/DataE&DAI/LabelsTxt"
outputPath = "/home/tom/Scripts&Data/DataE&DAI/TfRecords"
filenameListFile = "./training.txt"
label = b"person"
formatImg = b'jpg'

def getDataFile(labelFile, width, height):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        f = open(labelFile, "r")
        fLines = f.readlines()

        for line in fLines:
                datas = line.split()
                xmins.append(int(datas[1])/width)
                ymins.append(int(datas[2])/height)
                xmaxs.append(int(datas[3])/width)
                ymaxs.append(int(datas[4])/height)
                classes_text.append(label)
                classes.append(1)

        return xmins, ymins, xmaxs, ymaxs, classes_text, classes

def create_tf_example(imageFile, labelFile):
        image = Image.open(imageFile)
        width, height = image.size
        filename = bytes(imageFile.replace(imagesPath,"")[1:], 'utf-8')
        image_format = formatImg

        with tf.io.gfile.GFile(imageFile, 'rb') as fid:
                encoded_jpg = fid.read()
        encoded_image_data = encoded_jpg
        
        xmins, ymins, xmaxs, ymaxs, classes_text, classes = getDataFile(labelFile, width, height)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_image_data),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

def decode(raw_bytes):
    return tf.image.decode_jpeg(raw_bytes, channels=3, dct_method='INTEGER_ACCURATE')

def _read_from_tfrecord(example_proto):
        features = {
                'image/filename': tf.io.FixedLenFeature((), tf.string, default_value=''),
                'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
                'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value='')
        }
        ex = tf.io.parse_example([example_proto], features=features)

        image_tf = tf.map_fn(decode, ex['image/encoded'], dtype=tf.uint8, back_prop=False, parallel_iterations=10)
        filename_tf = ex['image/filename']
        bbox_tf = tf.stack([ex['image/object/bbox/%s' % x].values
                        for x in ['xmin', 'xmax', 'ymin', 'ymax']])
        bbox_tf = tf.transpose(tf.expand_dims(bbox_tf, 0), [0, 2, 1])

        return image_tf, filename_tf, bbox_tf

def verifInputs(name):
        print("Check ", name, ".tfrecords files", sep="")
        filename = outputPath + "/" + name + ".tfrecords"
        data_path = tf.compat.v1.placeholder(dtype=tf.string, name="tfrecord-file")
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(_read_from_tfrecord)
        iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(dataset), tf.compat.v1.data.get_output_shapes(dataset))
        image_tf, filename_tf, bbox_tf = iterator.get_next()
        iterator_init = iterator.make_initializer(dataset, name="dataset_init")

        with tf.compat.v1.Session() as sess:
                sess.run(iterator_init, feed_dict={data_path: filename})
                image_tf, filename_tf, bbox_tf = sess.run([image_tf, filename_tf, bbox_tf])
                width = len(image_tf[0][0])
                height = len(image_tf[0])
        for j in range(len(bbox_tf[0])):
                cv2.rectangle(image_tf[0], (int(bbox_tf[0][j][0]*width), int(bbox_tf[0][j][2]*height)), (int(bbox_tf[0][j][1]*width), int(bbox_tf[0][j][3]*height)), (0,255, 0), 2)
        plt.figure(num=filename_tf[0].decode("utf-8"))
        frameR = cv2.cvtColor(image_tf[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite(name + ".jpeg", frameR)

        

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
    serialized_example,
    features={
        'image/filename': tf.io.FixedLenFeature((), tf.string, default_value='')
    })
    filename_tf =  features['image/filename']
    return filename_tf

def extractFilename():
        listFilename = []
        with tf.compat.v1.Session() as sess:
                filename_queue = tf.train.string_input_producer([outputPath + "/test.tfrecords"])
                filename_tf = read_and_decode(filename_queue)
                init_op = tf.initialize_all_variables()
                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                for i in range(10000):
                        filename = sess.run([filename_tf])
                        listFilename.append(filename[0])
                coord.request_stop()
                coord.join(threads)
        listFilenameWithoutDuplicate = list(set(listFilename))
        listFilenameWithoutDuplicate.sort()
        listFile = [filename.decode("utf-8") for filename in listFilenameWithoutDuplicate]
        listName = [file[:-3] for file in listFile]
        strName = '\n'.join(listName)
        f = open(filenameListFile, "w")
        f.write(strName)
        f.close()

def main(_):
        filenameTrain = os.path.join(outputPath, 'train.tfrecords')
        filenameTest = os.path.join(outputPath, 'test.tfrecords')

        images = sorted(glob.glob(imagesPath + "/*.jpg"))
        labels = sorted(glob.glob(labelsPath + "/*.txt"))
        print(len(images), "images detected.")
        print(len(labels), "labels detected.\n")

        imagesAndLabels = list(zip(images, labels))
        random.shuffle(imagesAndLabels)
        images, labels = zip(*imagesAndLabels)
        
        ptrainSize = 80
        if len(sys.argv) > 1:
                if int(sys.argv[1]) < 0 or int(sys.argv[1]) > 100:
                        print("Error: argv[1] =", sys.argv[1], "> 100 or < 0")
                        exit(84)
                ptrainSize = int(sys.argv[1])
        print("Train size",  ptrainSize, "%\nTest size", 100-ptrainSize, "%\n")
        trainSize = int(ptrainSize * len(images) / 100)
        imagesTrain = images[:trainSize]
        imagesTest = images[trainSize:]
        labelsTrain = labels[:trainSize]
        labelsTest = labels[trainSize:]

        for i in range(len(labelsTest)):
                if (imagesTest[i][:-3].replace(imagesPath,"") != labelsTest[i][:-3].replace(labelsPath,"")):
                        print("Error: ", imagesTest[i], labelsTest[i])
                        exit(84)
        for i in range(len(labelsTrain)):
                if (imagesTrain[i][:-3].replace(imagesPath,"") != labelsTrain[i][:-3].replace(labelsPath,"")):
                        print("Error: ", imagesTrain[i], labelsTrain[i])
                        exit(84)
        
        writerTrain = tf.io.TFRecordWriter(filenameTrain)
        for j in range(len(labelsTrain)):
                if (j% 100 == 0):
                        print(j, "/", len(labelsTrain), end="\r")
                tf_example = create_tf_example(imagesTrain[j], labelsTrain[j])
                writerTrain.write(tf_example.SerializeToString())
        print(len(labelsTrain), "/", len(labelsTrain), "TRAIN SET DONE")
        writerTrain.close()

        writerTest = tf.io.TFRecordWriter(filenameTest)
        for j in range(len(labelsTest)):
                if (j% 100 == 0):
                        print(j, "/", len(labelsTest), end="\r")
                tf_example = create_tf_example(imagesTest[j], labelsTest[j])
                writerTest.write(tf_example.SerializeToString())
        print(len(labelsTest), "/", len(labelsTest), "TEST SET DONE\n")
        writerTest.close()

        print("COMPLETE\n")
        verifInputs('train')
        verifInputs('test')
        print("\nCOMPLETE")

if __name__ == "__main__":
        tf.compat.v1.app.run()