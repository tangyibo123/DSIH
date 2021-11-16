#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system and other libraries
import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps
import pandas as pd
import copy
import random
import glob
flatten = lambda l: [item for sublist in l for item in sublist]
class DataGeneratorTriplet(data.Dataset):
    def __init__(self, dataset, root, photo_dir, sketch_dir, photo_sd, sketch_sd,
                 #fls_sk, fls_imp, fls_imn, clss_sk,clss_p, clss_n,
                 batchsize, samples_per_class, transforms_sketch=None, transforms_image=None):
        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.photo_sd = photo_sd
        self.sketch_sd = sketch_sd
        '''
        self.fls_sk = fls_sk
        self.fls_imp = fls_imp
        self.fls_imn = fls_imn
        self.clss_sk = clss_sk
        self.clss_p = clss_p
        self.clss_n = clss_n
        '''
        self.batch_size = batchsize
        self.samples_per_class = samples_per_class
        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image
        # Load respective text-files
        #train = np.array(
         #   pd.read_table('/home/tangyibo/work/code/c_sketch/semtrans/src/tuberlin_mi.txt', header=None,
                   #       delim_whitespace=True))
        train = np.array(
            pd.read_table('/home/tangyibo/work/code/c_sketch/semtrans/src/sketchy_bigtrain.txt', header=None,
                         delim_whitespace=True))

        train_image_dict = {}
        for img_path, key, popath, nepath, cln in train:
            them = (img_path, popath, nepath, cln)
            if not key in train_image_dict.keys():
                train_image_dict[key] = []
            train_image_dict[key].append(them)
        self.image_dict = train_image_dict
        for sub in self.image_dict:
            newsub = []
            for instance in self.image_dict[sub]:
                newsub.append((sub, instance))
            self.image_dict[sub] = newsub

        # checks
        # provide avail_classes
        self.avail_classes = [*self.image_dict]
        self.reshuffle()

    def reshuffle(self):

        image_dict = copy.deepcopy(self.image_dict)
        for sub in image_dict:
            random.shuffle(image_dict[sub])

        classes = [*image_dict]
        random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >= self.samples_per_class) and (
                        len(batch) < self.batch_size / self.samples_per_class):
                    batch.append(image_dict[sub_class][:self.samples_per_class])
                    image_dict[sub_class] = image_dict[sub_class][self.samples_per_class:]

            if len(batch) == self.batch_size / self.samples_per_class:
                total_batches.append(batch)
                batch = []
            else:
                finished = 1

        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))
        print('smoothdata_len:{}'.format(len(self.dataset)))

    def __getitem__(self, item):
        #print(item)
        #print(len(self.dataset[item]))
        batch_item = self.dataset[item]
        clsk = batch_item[0]
        skpath = batch_item[1][0]
        popath = batch_item[1][1]
        nepath = batch_item[1][2]
        cln = batch_item[1][3]

        #print(clsk,skpath,popath)
        #print(cln,nepath)

        sk = ImageOps.invert(Image.open(skpath)).convert(mode='RGB')
        #sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item]))).\
         #   convert(mode='RGB')
        imp = Image.open(popath).convert(mode='RGB')
        imn = Image.open(nepath).convert(mode='RGB')
        #cls_sk = self.clss_sk[item]
        #cls_p = self.clss_p[item]
        #cls_n = self.clss_n[item]
        if self.transforms_image is not None:
            imp = self.transforms_image(imp)
            imn = self.transforms_image(imn)
        if self.transforms_sketch is not None:
            sk = self.transforms_sketch(sk)
            
        #skpath = os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item])

        return clsk, sk, clsk, imp, cln, imn, skpath

    def __len__(self):
        return len(self.dataset)



class DataGeneratorPaired(data.Dataset):
    def __init__(self, dataset, root, photo_dir, sketch_dir, photo_sd, sketch_sd, fls_sk, fls_im, clss,
                 transforms_sketch=None, transforms_image=None):
        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.photo_sd = photo_sd
        self.sketch_sd = sketch_sd
        self.fls_sk = fls_sk
        self.fls_im = fls_im
        self.clss = clss
        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image

    def __getitem__(self, item):
        sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item]))).\
            convert(mode='RGB')
        im = Image.open(os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])).convert(mode='RGB')
        cls = self.clss[item]
        if self.transforms_image is not None:
            im = self.transforms_image(im)
        if self.transforms_sketch is not None:
            sk = self.transforms_sketch(sk)
        return sk, im, cls

    def __len__(self):
        return len(self.clss)

    def get_weights(self):
        weights = np.zeros(self.clss.shape[0])
        uniq_clss = np.unique(self.clss)
        for cls in uniq_clss:
            idx = np.where(self.clss == cls)[0]
            weights[idx] = 1 / idx.shape[0]
        return weights


class DataGeneratorSketch(data.Dataset):
    def __init__(self, dataset, root, sketch_dir, sketch_sd, fls_sk, clss_sk, transforms=None):
        self.dataset = dataset
        self.root = root
        self.sketch_dir = sketch_dir
        self.sketch_sd = sketch_sd
        self.fls_sk = fls_sk
        self.clss_sk = clss_sk
        self.transforms = transforms

    def __getitem__(self, item):
        sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item]))).\
            convert(mode='RGB')
        cls_sk = self.clss_sk[item]
        if self.transforms is not None:
            sk = self.transforms(sk)
        skpath = os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item])

        return sk, cls_sk, skpath

    def __len__(self):
        return len(self.fls_sk)


class DataGeneratorImage(data.Dataset):
    def __init__(self, dataset, root, photo_dir, photo_sd, fls_im, clss_im, transforms=None):

        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.photo_sd = photo_sd
        self.fls_im = fls_im
        self.clss_im = clss_im
        self.transforms = transforms

    def __getitem__(self, item):
        im = Image.open(os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])).convert(mode='RGB')
        cls_im = self.clss_im[item]
        if self.transforms is not None:
            im = self.transforms(im)

        impath = os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])
        return im, cls_im, impath

    def __len__(self):
        return len(self.fls_im)
