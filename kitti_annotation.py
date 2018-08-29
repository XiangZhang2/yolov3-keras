
import os

classes = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]

def convert_annotation(image_id, list_file):
    #convert a label file
    in_file = open('/Users/xiang/Downloads/data_object_image_2/label_2/%s.txt'%(image_id))
    lines = in_file.readlines()
    for line in lines:
        line = line.split()
        if line[0] not in classes:
            continue
        cls_id = classes.index(line[0])
        box = (float(line[4]), float(line[5]), float(line[6]), float(line[7]))
        list_file.write(" " + ",".join([str(a) for a in box]) + ',' + str(cls_id))


path = '/Users/xiang/Downloads/data_object_image_2/training/image_2/'
filenames = os.listdir(path)
filenames.sort()
list_file = open('kitti_train.txt', 'w') #a new file
for name in filenames:
    name = name.split('.')
    image_id = str(name[0])
    list_file.write(path + image_id + '.png') #image_id with path
    convert_annotation(image_id, list_file)
    list_file.write('\n')










