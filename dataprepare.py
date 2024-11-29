from pyexpat import features
import torch
import pandas as pd
import pathlib
import os
import itertools
import torchvision.datasets
from bs4 import BeautifulSoup
from PIL import Image
from mpl_toolkits.mplot3d.proj3d import transform
from tzdata import IANA_VERSION
import numpy as np
from torch.utils.data import Dataset, DataLoader
import supervision as sv
import cv2
from torch.utils.data import dataloader
import splitfolders
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2



class myDataset(Dataset):
    def __init__(self,features,labels,transform=None):
        super().__init__()
        print("Initializing Dataset")
        self.features = features
        self.labels = labels
        self.transform = transform



    def __len__(self):
        print(f"Length of features {len(self.features)}")
        print(f"Length of labels {len(self.labels)}")
        return len(self.labels)

    def __getitem__(self,idx):
        features_iter = self.features[idx]
        labels_iter = self.labels[idx]

        features_iterated = self.transform(features_iter)
        labels_iterated = torch.tensor(labels_iter)

        print(f"Item idx {idx} "
              f"Labels list {labels_iterated}")


        return features_iterated,labels_iterated

    def show_attributes(self):
        print(f"Images {self.images}")
        print(f"Labels {self.labels}")


class Datapreparer():
    def __init__(self,data_path):
        data_path = pathlib.Path(data_path)
        self.image_directory = data_path.joinpath('VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages')
        self.anotations_directory = data_path.joinpath('VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations')
        self.imagesets_directory = data_path.joinpath('VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets')
        self.segmentation_object = data_path.joinpath('VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/SegmentationObject')
        self.segmentation_directory = data_path.joinpath('VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/SegmentationClass')

        # TESTING DATA
        self.image_directory_test = data_path.joinpath('VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages')
        self.anotations_directory_test = data_path.joinpath('VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations')
        self.imagesets_directory_test = data_path.joinpath('VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets')
        self.sementation_object = data_path.joinpath('VOCtest_06-Nov-2007/VOCdevkit/VOC2007/SegmentationObject')
        self.segmentation_directory_test = data_path.joinpath('VOCtest_06-Nov-2007/VOCdevkit/VOC2007/SegmentationClass')
        self.dict_equivalence = {
            "aeroplane" : 1,
            "bicycle" : 2,
            "bird" : 3,
            "boat" : 4,
            "bottle" : 5,
            "bus" : 6,
            "car" : 7,
            "cat" : 8,
            "chair" : 9,
            "cow" : 10,
            "diningtable" : 11,
            "dog" : 12,
            "horse" : 13,
            "motorbike" : 14,
            "person" : 15,
            "pottedplant" : 16,
            "sheep" : 17,
            "sofa" : 18,
            "train" : 19,
            "tvmonitor" : 20
        }





    def extract_info_xml(self,trainbool):
        document_dict = {}
        if trainbool:
            file_list = os.listdir(self.anotations_directory)
            anotation = self.anotations_directory
            print(f"Since training is true xml file route is {anotation}")
        else:
            file_list = os.listdir(self.anotations_directory_test)
            anotation = self.anotations_directory_test
            print(f"Since training is false xml file route is {anotation}")

        for file in file_list:
            document_dict[file] = []
            new_path = os.path.join(anotation, file)
            try:
                with open(new_path,'r') as f:
                    data = f.read()
                    bs_data = BeautifulSoup(data, 'lxml-xml')
                    bs_filename = bs_data.find('filename').text
                    bs_objects = bs_data.find_all('object')

                    obj_number = 0
                    for obj in bs_objects:
                        obj_number+=1
                        bs_x_min = obj.find('xmin').text
                        bs_y_min = obj.find('ymin').text
                        bs_x_max = obj.find('xmax').text
                        bs_y_max = obj.find('ymax').text
                        object_name = obj.find('name').text
                        object_name = self.dict_equivalence[object_name]
                        new_list =  [float(bs_x_min),float(bs_y_min), float(bs_x_max), float(bs_y_max),float(object_name)]
                        document_dict[file].append(new_list)

            except Exception as error:
                print(f"Error while reading xml file check it's sintax: {error}")
        #CALL NEXT FUNCTION USING DOCUMENT DICT
        data_dict = self.get_images_folder(document_dict)
        image_list = self.get_image_with_data(data_dict,trainbool)
        image,bboxes = self.get_image_bboxes(data_dict, trainbool)
        #bboxes = bbox_iterator(bboxes)
        return image,bboxes


    def get_images_folder(self,data_dict):
        images_names,values = zip(*data_dict.items())
        images_names = list(images_names)
        values = list(values)


        position = 0
        for image in list(images_names):
            image = image.replace('.xml', '.jpg')
            images_names[position] = image
            position+=1
        data_dict = dict(zip(images_names, values))


        return data_dict

    def get_image_with_data(self,data_dict,train):
        image_list = []

        image_names = data_dict.keys()
        image_names = list(image_names)
        if train:
            image_path = self.image_directory
            print(f"Since training is true image_path is {image_path}")
        else:
            image_path = self.image_directory_test
            print(f"Since test is true image_path is {image_path}")

        for image in image_names:

            new_image = image_path.joinpath(image)

            try:
                new_image = Image.open(new_image)
                new_image = np.array(new_image)
                image_list.append(new_image)

            except Exception as error:
                print(f"Image {image} failed at loading")
                print(f"The route we tried to opne is {image_path}")
                break

        return image_list
    def image_show_fn(self,document_dict,image_folder):
        #REMEMBER THAT WE REMOVED TOTAL OBJECTS FROM EXTRACT_INFO_XML SO THIS FUNCTION WONT WORK
        images,bbox = zip(*document_dict.items())
        images = list(images)
        bbox = list(bbox)
        bbox_list = bbox[1]
    #If you want to initialize the array length to the total of objects in the image
    #total_objects = bbox_list[0][0]
        anotations_array = np.empty((0,4))
        every_object = []
        for bbox in bbox_list:
            print(f"Bbox for the image on: {images[0]}")

            x1 = int(bbox[2])
            y1 = int(bbox[3])
            x2 = int(bbox[4])
            y2 = int(bbox[5])

            total_objects = int(bbox[0])
            coord_array = np.array([[x1,y1,x2,y2]])

            print(f"Bounding box coordinate x1 {x1} coordinate x2 {x2} coordinate y1 {y1} coordinate y2 {y2}")
            image = os.path.join(image_folder, images[1])
            image = cv2.imread(image)

            anotations_array = np.append(anotations_array,coord_array,axis=0)
            array_confidence = np.random.uniform(0.50,0.95, total_objects)

        print(f"Every object name {every_object}")
        array_ids = np.array([i for i in range(0,total_objects)])
        detections = sv.Detections(
            xyxy = anotations_array,
            class_id=array_ids,
            confidence=array_confidence
        )

        bounding_box_anotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        anotated_frame = bounding_box_anotator.annotate(
            scene = image.copy(),
            detections = detections,

        )
        anotated_frame = label_annotator.annotate(
            scene = anotated_frame,
            detections = detections,
            labels = every_object
        )
        sv.plot_image(anotated_frame)

    def data_transformations(self):
        train_transformations = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        bbox_transformations = transforms.ToTensor()

        test_transformations = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_transformations, test_transformations

    def get_image_bboxes(self,data_dict,train):
        images,bboxes = zip(*data_dict.items())
        images = list(images)
        bboxes = list(bboxes)
        images_list = []
        position = 0
        if train:
            image_path = self.image_directory
        else:
            image_path = self.image_directory_test
        for image in images:
            new_path = os.path.join(image_path, image)
            image = Image.open(new_path)
            images_list.append(image)
        return images_list, bboxes

def bbox_iterator(bboxes):
    new_bboxes = []
    for bbox in bboxes:
        new_bbox = list(itertools.chain.from_iterable(bbox))
        new_bboxes.append(new_bbox)
    return new_bboxes

def controller(train_data):
    for X,y in tqdm(train_data, desc = "Training"):
        print(f"X para el tqdm {X}")
        print(f"Y para el tqdm {y}")





