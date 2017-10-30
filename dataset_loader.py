import torch
import torch.utils.data
import os
from os.path import isfile
from xml.etree import ElementTree as ET
from PIL import Image
import numpy as np

# Using the VOC2007 dataset provided for assignment 2
classes = ('car','bicycle',
           'aeroplane', 'cat', 'dog', 'bird', 'boat',
           'bottle', 'bus', 'chair',
           'cow', 'diningtable', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')



class create_dataset(torch.utils.data.Dataset):
    def __init__(self,train,transform=None):
        
        self.train = train
        self.transfrom = transform

        if self.train:
            self.images_path = './VOCtrain/VOC2007/JPEGImages'
            self.annotations_path = './VOCtrain/VOC2007/Annotations'
        else:
            self.images_path = './VOCtest/VOC2007/JPEGImages'
            self.annotations_path = './VOCtest/VOC2007/Annotations'

        self.data = []

        images = sorted(os.listdir(self.images_path))
        annotations = sorted(os.listdir(self.annotations_path))


        for img, annot in zip(images, annotations):
            name1 = img.split('.')[0]
            name2 = img.split('.')[0]

            # making sure image and annotation file are same.

            assert(name1 == name2)

            #  parsing annotation xml
            tree = ET.parse(os.path.join(self.annotations_path, annot))
            root = tree.getroot()

            # list storing all objects present in an image.
            objects = []
            for child in root:
                if child.tag == 'object':
                    # print(child[0].text)
                    objects.append(child[0].text)
            
            # finding unique class present in an image
            objects = set(objects)

            # print(objects)

            # giving multiple labels to images.
            labels = np.zeros(len(classes))

            # using each unique class as label for image.
            for ob in objects:
                # using class index as class label
                class_label = classes.index(ob)
                labels[class_label] = 1
                # self.data.append((os.path.join(self.images_path, img), class_label))
            
            labels = torch.Tensor(labels)
            self.data.append((os.path.join(self.images_path, img), labels))
    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self,idx):
        image = Image.open(self.data[idx][0])
        # print(self.data[idx][1])
        return (self.transfrom(image), self.data[idx][1])




# dataset = create_dataset(train=True)
# print(len(dataset))

def create_data_loader(batch_size, train, transform=None):
    dataset = create_dataset(train=train, transform=transform)

    return (torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True), len(dataset))
