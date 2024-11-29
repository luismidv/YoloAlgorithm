import torch
from PIL.TiffTags import TYPES
from jinja2.optimizer import Optimizer
from matplotlib.colors import cnames
from sympy import print_tree
from torch import nn
import dataprepare as dp
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

#NEURAL NETWORK SCHEMA
    #24 CONV LAYERS
    #2 FC LAYERS
    #1X1 REDUCTION LAYER FOLLOWED BY 3X3 CONV
    #PRE TRAIN ON 256*256 DOUBLE AFTER FOR DETECTION
    #OUTPUT AS 7*7*30

    #PRE TRAINING SHOULD BE MADE IN 20 FIRST CONV LAYERS
    #RUN 135 EPOCHS ON A BATCH = 64

class  CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size = 3,stride = 1,padding = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding = padding , stride = stride)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        return x
class YoloNeuralNet(nn.Module):
    def __init__(self,parameters):
        super().__init__()
        num_channels, height,width = parameters['initial_sizes']
        num_classes = parameters['num_classes']

        self.yoloarch = nn.ModuleList([

            CNNBlock(num_channels, 64, kernel_size=7, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            CNNBlock(64, 192, kernel_size=7, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            CNNBlock(192, 128, kernel_size=1, stride=1, padding=1),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            CNNBlock(256, 256, kernel_size=1, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 512, kernel_size=1, stride=1, padding=1),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            CNNBlock(1024, 256, kernel_size=1, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 512, kernel_size=1, stride=1, padding=1),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=1),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=1),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=2, padding=1),


        ])

    def forward(self,X):
        num_linear = 0
        for layer in self.yoloarch:
            X = layer(X)

        X = X.view(X.size(0), -1)
        X = nn.Linear(102400,4096)(X)
        X = nn.ReLU(inplace=True)(X)
        X = nn.Dropout(0.5)(X)
        X = nn.Linear(4096,7*7*30)(X)
        X = F.sigmoid(X)
        X = X.view(30,7,7)

        return X

def accuracy_calculator(prediction, label):
    top_pred = prediction.argmax(1, keepdim=True)
    correct = top_pred.eq(label.view_as(top_pred)).sum()
    acc = correct.float()/label.shape[0]
    return acc





def model_training2(model,dataloader,device, loss_fn, optimizer):
    model_accuracy = []
    model_loss = []
    count = 0
    for X,Y in tqdm(dataloader):
        #cell_responsible = calculate_cell_for_detection(Y)
        pred_dict = {}
        X = X.to(device)
        Y = Y.to(device)

        model.train()
        optimizer.zero_grad()
        prediction = model(X)
        prediction = non_max_superssion(prediction,0.5)
        print(f"prediction {prediction.shape}")

        bbox_list = grid_determination(prediction,Y)
        for i in range(len(bbox_list)):
            pred_dict[i] = (bbox_list[i][5],bbox_list[i][6])
        print(pred_dict)
        keys = list(pred_dict.keys())
        for i in pred_dict.keys():
            pixels = list(pred_dict[i])
            prediction_toloss = prediction[:,int(pixels[0]), int(pixels[1])]
            bbox1 = prediction_toloss[1:5]
            bbox2 = prediction_toloss[6:10]
            usefull_bbox = inter_over_union(bbox1,bbox2,Y[0][i])
            print(f"Prediction shape {usefull_bbox}")
            print(f"Gr shape {Y[0][i]}")
            loss = F.mse_loss(usefull_bbox, Y[0][i][0:-1])
            loss.backward()
            optimizer.step()
            acc = accuracy_calculator(prediction, Y)
            return acc, loss

        break

def model_training(model,dataloader,device, loss_fn, optimizer):
    model_accuracy = []
    model_loss = []
    count = 0
    for X,Y in tqdm(dataloader):
        #cell_responsible = calculate_cell_for_detection(Y)
        pred_dict = {}
        X = X.to(device)
        Y = Y.to(device)

        model.train()
        prediction = model(X)
        item_list = grid_determination(prediction,Y)
        #prediction = non_max_superssion(prediction,0.5)
        print(f"prediction {prediction[1:5,1,1].shape}")
        print(f"Gr shape {Y[0][0][:-1].shape}")
        #bbox_list = grid_determination(prediction,Y)
        keys = list(pred_dict.keys())
        item_num = 0
        total_loss = 0

        for item in item_list:
            print(f"Numero de objetos {len(item_list)} | Iteraccion numbero {item_num}")
            loss = F.mse_loss(prediction[1:5,int(item[0]),int(item[1])], Y[0][item_num][:-1])
            total_loss += loss
            item_num = item_num + 1
        optimizer.zero_grad()
        total_loss.backward()
        print(total_loss)
        optimizer.step()

        







def grid_determination(prediction,y):
    print(f"Label {y}")
    print(f"Label length{y.shape}")
    item_list = []
    for i in range(y.shape[1]):
        item_list.append((y[0][i][0],y[0][i][1],y[0][i][2],y[0][i][3],y[0][i][4]))
    new_item_list = []
    for item in item_list:

        x_center = (item[2]+item[0])/2
        y_center = (item[3]+item[1])/2

        x_pixel = (x_center/64) - 1
        y_pixel = (y_center/64) - 1
        pixel = (np.floor(x_pixel),np.floor(y_pixel))

        new_item_list.append(pixel)
    return new_item_list

def get_data_from_prediction(prediction):
    prediction = prediction.view(30,7,7)
    first_pixel_data = prediction[:, 0,0]
    second_pixel_data = prediction[:, 0,1]
    class_probabilities = prediction[20:30,0,0 ]

    print(f"First pixel data class probabilities: {class_probabilities}")
    #print(f"First pixel data {second_pixel_data}")

def non_max_superssion(prediction,threshold):
    total_bbox = 98
    for i in range(prediction.shape[1]):
        for j in range(prediction.shape[2]):
            confidence_bbox1 = prediction[0,i,j]
            confidence_bbox2 = prediction[5,i,j]

            if confidence_bbox1 < threshold:
                total_bbox = total_bbox - 1
                prediction[1:5,i,j] = 0
                bbox_printing(prediction,i,j,1)

            if confidence_bbox2 < threshold:
                total_bbox = total_bbox - 1
                prediction[6:10,i,j] = 0
                bbox_printing(prediction,i,j,2)
    return prediction
def bbox_printing(prediction,i,j,bbox):
    if bbox == 1:
        pos = 0
    else:
        pos = 5
    print(f"Confianza en la bbox 1 del pixel {prediction[pos, i, j]} \n"
          f"Coordenada x1 cae en la bbox {7 * prediction[pos+1, i, j]} \n"
          f"Coordenada y1 cae en la bbox {7 * prediction[pos+2, i, j]} \n"
          f"Coordenada x2 cae en la bbox {7 * prediction[pos+3, i, j]} \n"
          f"Coordenada y2 cae en la bbox {7 * prediction[pos+4, i, j]} \n"

          )
def inter_over_union_helper(bbox1,ground_truth_coordinates):

    A_inter = max(ground_truth_coordinates[0], bbox1[0])
    B_inter = max(ground_truth_coordinates[1], bbox1[1])
    C_inter = min(ground_truth_coordinates[2], bbox1[2])
    D_inter = min(ground_truth_coordinates[3], bbox1[3])
    print(f"Parametros recibidos {bbox1} \n"
          f"{ground_truth_coordinates}")

    #if C_inter < A_inter or D_inter < B_inter:
        #print(f"Error en las restricciones"
              #f"{C_inter} | {A_inter} | {B_inter} | {D_inter}")
        #iou = 0
        #return iou

    box_area = (C_inter - A_inter) * (D_inter - B_inter)
    first_box = (ground_truth_coordinates[2] - ground_truth_coordinates[0]) * (
                ground_truth_coordinates[3] - ground_truth_coordinates[1])
    second_box = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

    union = first_box + second_box - box_area
    inter_over_union = box_area / union
    return inter_over_union

def inter_over_union(bbox1,bbox2,ground_truth):
    #DEFINE BBOX COORDINATES

    bbox_with_iou = []
    print(f"Gr {ground_truth} ")
    bbox1_iou = inter_over_union_helper(bbox1,ground_truth)
    bbox2_iou = inter_over_union_helper(bbox2,ground_truth)

    if bbox1_iou > bbox2_iou:
        print(f"Caja 1 es mejor que caja 2 {bbox1_iou}")
        return bbox1
    else:
        print(f"Caja 2 es mejor que caja 1 {bbox2_iou} ")

        return bbox2





data_prepare = dp.Datapreparer("./cfg/archive")
train_transform, test_transform = data_prepare.data_transformations()
#data_treat_check(image_directory, anotations_directory)
images,bboxes = data_prepare.extract_info_xml(trainbool = True)
print(f"Bboxes {bboxes[0]}")
print("XML train read finished")
images_test,bboxes_test = data_prepare.extract_info_xml(trainbool = False)
print("XML test read finished")
print(len(bboxes))
train_data = dp.myDataset(images,bboxes,transform=train_transform)
#test_data = dp.myDataset(images_test,bboxes_test,transform=test_transform)
train_loader = DataLoader(train_data, batch_size=1, shuffle = True)
#test_loader = DataLoader(test_data, batch_size=64, shuffle = True)

parameters = {
    'initial_sizes' : (3,448,448),
    'num_classes': 20
}

device = torch.device("cpu")
myYoloNet = YoloNeuralNet(parameters)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(myYoloNet.parameters(), lr=3e-4)
prediction = model_training(myYoloNet, train_loader, device, loss_fn, optimizer)
#get_data_from_prediction(prediction)
#non_max_superssion(prediction,0.5)
#prediction = prediction.view(30,7,7)
#iou_list = inter_over_union(prediction,bboxes)
##print(f"IOU {iou_list}")




