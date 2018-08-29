# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    _defaults = {
        #"model_path": 'model_data/yolo_weights.h5',
        "model_path": '/data/output/weights/balencedata_ep10_loss28,24_val_loss27.92.h5',
        #"anchors_path": 'model_data/yolo_anchors.txt',
        "anchors_path": '/data/code/model_data/yolo_anchors.txt',
        #"classes_path": 'model_data/coco_classes.txt', 
        "classes_path": '/data/code/model_data/kitti_classes.txt',
        ##因为coco的种类比较多，而且yolo.weigths也是训练coco的数据得来的
        "score" : 0.2,
        "iou" : 0.35,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session() #类似tensorflow定义一个sess吧
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        #返回所有class的列表
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        #返回所有的anchor，shape=(9,2)，type:np.array
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        #核心还是调用eval函数生成box和score
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors) #9
        num_classes = len(self.class_names) #80
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False) #如果try没有抛出错误，except就不会执行
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
            #当try异常时，会执行except
        else:
            print('load_model success') #自己加的
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
            #如果try正常，不except，会else，如果try抛出错误，会except，不else

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.) #这里应该加float，不然都是0
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors)) #self.color后面会用到，
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, )) #why placeholder?
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        #下面才是这个函数的核心表达，只是调用了eval函数
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes
        #返回的结果和evl一样

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32') #变成float

        # print(image_data.shape)
        image_data /= 255. #归一化
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        out_boxes = out_boxes #这个就不应该+1了
        box_yx = out_boxes[:,0:2]
        box_hw = out_boxes[:,2:4]
        box_xy = box_yx[:,::-1]
        box_wh = box_hw[:,::-1]
        out_boxes[:,0:2] = box_xy
        out_boxes[:,2:4] = box_wh
        #把(x,y,w,h)的box变为(left_top,right_low),并且左上角的坐标为1
        return out_boxes, out_scores, out_classes


def get_classlist(class_path):
    #已验证
    with open(class_path) as f:
        lines = f.readlines()
        class_list = [x.strip() for x in lines]
    return class_list


def listdir(path):
    #读取文件夹下除DS_store以外的所有文件名到list
    images_name = os.listdir(path)
    images_name.sort()
    if images_name[0] == '.DS_Store':
        images_name = images_name[1:]
    return images_name


def image_name(image_path, file_dir, file_name):
    #用来生成包含所有测试图片name的txt文件
    images_name = listdir(image_path)
    with open(file_dir + file_name, 'w') as f:
        for i in range(len(images_name)):
            f.write(str(images_name[i][: -4]) + '\n')


def get_dir(data_name):
    
    if data_name == 'VOC2007':
        image_file = '/Users/xiang/Downloads/keras-yolo3/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
        iamge_path = '/Users/xiang/Downloads/keras-yolo3/VOCdevkit/VOC2007/JPEGImages/'
        det_path = '/Users/xiang/Downloads/keras-yolo3/VOCdevkit/VOC2007/pascal_dets/'
        classes_file = '/Users/xiang/Downloads/keras-yolo3/model_data/coco_classes.txt'
        image_format = '.txt'
    
    if data_name == 'kitti':
        image_file = '/data/val/val.txt'
        image_path = '/Users/xiang/Downloads/data_object_image_2/data/val/'
        det_path = '/data/val/11epoch_with_balence_data_val/'
        classes_file = '/data/code/model_data/kitti_classes.txt'
        image_format = '.png'

    else:
        print('No data_name match！')

    return image_file, image_path, det_path, classes_file, image_format




def eval(image_file, image_path, det_path, classes_file, image_format):
    '''
    生成用于计算mAP的dets文件
    image_file:所有输入文件名的txt文件
    path:测试图像路径
    det_path:存放测试结果的path路径
    classes_file:存放所有class_name的txt文件
    '''
    yolo = YOLO()
    file = open(image_file)
    imagenames = file.readlines()
    imagenames = [imagename.strip() for imagename in imagenames]
    imagenames_with_path = [image_path + imagename + image_format for imagename in imagenames]

    class_list = get_classlist(classes_file)
    out_list = []

    #测试每一张图片
    for i in range(len(imagenames_with_path)):
        image = Image.open(imagenames_with_path[i])
        out_boxes, out_scores, out_classes = yolo.detect_image(image)
        print(imagenames[i])
        #对于每一张图片，将得到的结果输出到out_list中
        for j in range(len(out_classes)):
            result = [out_classes[j], imagenames[i],str(out_scores[j]), str(out_boxes[j,0]),
                      str(out_boxes[j,1]), str(out_boxes[j,2]), str(out_boxes[j,3])]
            out_list.append(result)

    with open(det_path + 'all.txt', 'w') as f:
        for a in range(len(out_list)):
            f.write(str(out_list[a][0]) + ' ' + out_list[a][2] + ' ' + out_list[a][3] + ' ' + out_list[a][4] + ' ' + out_list[a][5] + ' ' + out_list[a][6])
            f.write('\n')
    print('Save all succeed!!!')

    #按照class分别保存结果，为计算mAP准备
    det_file = {}
    for classname in class_list: #一个类别一个类别写入
        det_file[classname] = open(det_path + classname + '.txt', 'w')
        for k in range(len(out_list)):
            if class_list[out_list[k][0]] == classname:
                det_file[classname].write(out_list[k][1] + ' ' + out_list[k][2] + ' ' +
                                          out_list[k][3] + ' ' + out_list[k][4] + ' ' +
                                          out_list[k][5] + ' ' + out_list[k][6])
                det_file[classname].write('\n')
        det_file[classname].close()
    print('Save each succeed!!!')





























