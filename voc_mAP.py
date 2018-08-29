# -*— coding:utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""

import cPickle
import logging
import numpy as np
import os
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def kitti_rec(annotation_path):
    recs = {}
    classes = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram','Misc','Doncare']
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            line = line.split(' ')
            imagename = line[0][-10:-4]
            objects = []
            for i in range(1, len(line)):
                obj_struct = {}
                line[i] = line[i].split(',')
                obj_struct['name'] = classes[int(line[i][4])]
                obj_struct['difficult'] = 0
                obj_struct['bbox'] = [float(line[i][0]),float(line[i][1]),
                                      float(line[i][2]),float(line[i][3])]
                objects.append(obj_struct)
            recs[imagename] = objects
        return recs


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0 
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]) #计算面积
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             kitti=True
             ):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    imageset = os.path.splitext(os.path.basename(imagesetfile))[0]
    cachefile = os.path.join(cachedir, imageset + '_annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    #到下一个注释前的代码，将标签文件中的信息读取到一个字典里，
    #格式为{'imagename':[parse_rec().output]······},包含所有标签
    if not os.path.isfile(cachefile):
        # load annots
        if not kitti:
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = parse_rec(annopath.format(imagename))
                if i % 100 == 0:
                    logger.info(
                        'Reading annotation for {:d}/{:d}'.format(
                            i + 1, len(imagenames)))
        else:
            recs = kitti_rec(annopath)
        # save
        logger.info('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname] 
                bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R) #这个值是用来判断是否重复检测的
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence 
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids) 
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf 
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]: 
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1 #判断是否重复检测，检测过一次以后，值就从False变为1了
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp) 
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def get_classlist(class_path):

    with open(class_path) as f:
        lines = f.readlines()
        class_list = [x.strip() for x in lines]
    return class_list


def get_dir(data_name):
    
    if data_name == 'VOC2007':
        detpath = '/Users/xiang/Downloads/keras-yolo3/VOCdevkit/VOC2007/pascal_dets/{}.txt'
        annopath = '/Users/xiang/Downloads/keras-yolo3/VOCdevkit/VOC2007/Annotations/{}.xml'
        imagesetfile = '/Users/xiang/Downloads/keras-yolo3/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
        cachedir = '/Users/xiang/Downloads/keras-yolo3/cached/'
        class_path = '/Users/xiang/Downloads/keras-yolo3/model_data/voc_classes.txt'
    
    if data_name == 'kitti':
        detpath = '/data/val/90epoch_val_iou0.45/{}.txt'
        annopath = '/Users/xiang/Downloads/data_object_image_2/label/kitti_val.txt'
        imagesetfile = '/data/val/val.txt'
        cachedir = '/data/val/90epoch_val_iou0.45/cached/'
        class_path = '/data/code/model_data/kitti_classes.txt'

    else:
        print('No data_name match！')

    return detpath,annopath,imagesetfile,cachedir,class_path


def mAP():

    detpath,annopath,imagesetfile,cachedir,class_path = get_dir('kitti')
    ovthresh=0.3,
    use_07_metric=False

    rec = 0; prec = 0; mAP = 0
    class_list = get_classlist(class_path)
    for classname in class_list:
        rec, prec, ap = voc_eval(detpath,
                                 annopath,
                                 imagesetfile,
                                 classname,
                                 cachedir,
                                 ovthresh=0.5,
                                 use_07_metric=False,
                                 kitti=True)
        print('on {}, the ap is {}, recall is {}, precision is {}'.format(classname, ap, rec[-1], prec[-1]))
        mAP += ap
    
    mAP = float(mAP) / len(class_list)

    return mAP


















































