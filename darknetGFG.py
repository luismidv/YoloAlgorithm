import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
class Dataset(torch.utils.data.Dataset):
    def __init__(self,csv_file, image_dir, label_dir,anchors,
                    image_size = 416, grid_sizes = [13,26,52],
                    num_classes = 20, transform = None
    ):
        self.label_list = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.image_size = image_size
        self.transform = transform
        self.grid_sizes = grid_sizes

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])

        self.num_anchors = self.anchors.shape[0]

        self.num_anchors_per_scale = self.num_anchors // 3

        self.num_classes = num_classes
        self.ignore_iou_thres = 0.5

    def __len__(self):
        return len(self.label_list)
    
    def _getitem__(self,idx):
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx,1])
        #APPLY ROLL MOVING CLASS LABEL TO LAST COLUMN
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter = " ", ndmin=2),4,axis=1).tolist()

        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx,0])
        image = np.array(Image.open(img_path).convert("RGB"))

        #data augmentations
        if self.transform:
            augs = self.transform(image = image, bboxes = bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        targets = [torch.zeros((self.num_anchors_per_scale,s,s,6)) for s in self.grid_sizes]
        
        
        for box in bboxes:
            #IOU OF THE BOUNDING BOXES WUTG ABCGIR BIXES
            iou_anchors = inter_over_union(torch.tensor(box[2:4]),self.anchors, is_pred=False)

            #SELECT THE BEST ANCHOR BOX
            anchor_indices = iou_anchors.argsort(descending = True, dim = 0)
            x,y,width,height,class_label = box

            has_anchor = [False] * 3


            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                #IDENTIFYING THE GRID SIZE FOR THE SCALE
                s = self.grid_sizes[scale_idx]

                #IDENTIFYING THE CELL TO WHICH BOUNDING BOX BELONGS
                i,j = int(s*y), int(s+x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                #CHECK IF THE ANCHOR IS ALREADY ASIGNED
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale,i,j,0] = 1
                    #CALCULATING THE CENTER OF THE BOUNDING BOX RELATIVE TO THE CELL
                    x_cell, y_cell = s * x - j, s * y - i

                    #Calculating width and height of the bbox relative to the cell
                    width_cell, height_cell = (width*s, height*s)

                    #IDENTIFY THE BOX COORDINATES
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    #Asign box coordinates to the target
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates

                    #Asign the class label to the target
                    targets[scale_idx][anchor_on_scale,i,j,5] = int(class_label)

                    #Set the anchor box as assigned for the scale
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thres:
                    targets[scale_idx][anchor_on_scale,i,j,0] = -1
        return image,tuple(targets)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm = True, **kwargs):
        super(). __init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm
    
    def forward(self, x):
        x = self.conv(x)

        if self.use_batch_norm:
            x = self.bn(x)
            return self.activation(x)
        else:
            
            return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual = True, num_repeats = 1):
        super(). __init__()

        res_layers = []
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    nn.Conv2d(channels, channels//2, kernel_size=1),
                    nn.BatchNorm2d(channels//2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels//2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1)
                )
            ]
        
        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self,x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            if self.use_residual:
                x = x + residual
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(). __init__()

        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2*in_channels, (num_classes+5)*3, kernel_size=1)
        ),
        self.num_classes = num_classes

    def forward(self,x):
        output = self.pred(x)
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3))
        output = output.permute(0,1,3,4,2)
        return output

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20): 
        super().__init__() 
        self.num_classes = num_classes 
        self.in_channels = in_channels 
  
        # Layers list for YOLOv3 
        self.layers = nn.ModuleList([ 
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1), 
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(64, num_repeats=1), 
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(128, num_repeats=2), 
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(256, num_repeats=8), 
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(512, num_repeats=8), 
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1), 
            ResidualBlock(1024, num_repeats=4), 
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1), 
            ResidualBlock(1024, use_residual=False, num_repeats=1), 
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
            ScalePrediction(512, num_classes=num_classes), 
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
            nn.Upsample(scale_factor=2), 
            CNNBlock(768, 256, kernel_size=1, stride=1, padding=0), 
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1), 
            ResidualBlock(512, use_residual=False, num_repeats=1), 
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
            ScalePrediction(256, num_classes=num_classes), 
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0), 
            nn.Upsample(scale_factor=2), 
            CNNBlock(384, 128, kernel_size=1, stride=1, padding=0), 
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1), 
            ResidualBlock(256, use_residual=False, num_repeats=1), 
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0), 
            ScalePrediction(128, num_classes=num_classes) 
        ])
    def forward(self,x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer,ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)

            if isinstance(layer,ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim = 1)
                route_connections.pop()
        return outputs


def inter_over_union(box1,box2,is_pred = True):
    if is_pred:
        #Box coordinates of predicition
        b1_x1 = box1[...,0:1] - box1[...,2:3]/2
        b1_y1 = box1[...,1:2] - box1[...,3:4]/2
        b1_x2 = box1[...,0:1] - box1[...,2:3]/2
        b1_y2 = box1[...,1:2] - box1[...,3:4]/2

        #Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] - box2[..., 3:4] / 2

        #Getting coordinates of the intersenction rectangle
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.max(b1_x2, b2_x2)
        y2 = torch.max(b1_y2, b2_y2)

        #Make sure the intersection is at least 0
        intersection = (x2 -x1).clamp(0)*(y2-y1).clamp(0)

        box1_area = abs((b1_x2 - b1_x1) * (b1_y2-b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2-b2_y1))
        union = box1_area + box2_area - intersection

        #Calculate IoU
        epsilon = 1e-6
        iou_score = intersection/(union + epsilon)
        return iou_score
    else:
        intersection_area = torch.min(box1[...,0], box2[...,0]* \
                            torch.min(box1[...,1],box2[...,1]))

        box1_area = box1[...,0]*box1[...,1]
        box2_area = box2[...,0]*box2[...,1]
        union_area = box1_area + box2_area - intersection_area

        iou_score = intersection_area / union_area

        return iou_score
    
def non_maximum_supression(bboxes,iou_threshold,threshold):
    #Filter the boxes that recieved  if the threshold(trust) is below 
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key = lambda x: x[1], reverse=True)
    bboxes_nms = []

    while bboxes:
        #GET FIRS BBOX
        first_box = bboxes.pop(0)
        
        #ITERATE OVER THE REST
        for boxes in bboxes:
            if boxes[0] != first_box[0] or inter_over_union(
                torch.tensor(first_box[2:]),torch.tensor(boxes[2:]),) < iou_threshold:
                if boxes not in bboxes_nms:
                    bboxes_nms.append(boxes)
    return bboxes_nms

def convert_cell_to_bboxes(predictions, anchors, s, is_prediction = True):

    #BATCH SIZE
    batch_size = predictions.shape[0]
    #number of anchors
    num_anchors = len(anchors)
    #list of predictions
    box_predictions = predictions[...,1:5]

    #IF IS_PREDICTION THEN WE PASS THE X AND Y COORDINATE THROUGHT SIGMOID FUNCTION AND WIDTH AND HEIGHT TO EXPONENT
    if is_prediction:
        anchors = anchors.reshape(1, len(anchors),1,1,2)
        box_predictions[...,0:2] = torch.sigmoid(box_predictions[...,0:2])
        box_predictions[...,2:] = torch.exp(box_predictions[...,2:])*anchors
        scores = torch.sigmoid(predictions[...,0:1])
        best_class = torch.argmax(predictions[..., 5:], dim = .1).unsqueeze(-1)
    else:
        #JUST CALCULATE SCORES AND BEST CLASS 
        scores = predictions[...,0:1]
        best_class = predictions[...,5:6]
    #CALCULATE CELL INDICES
    cell_indices = (torch.arange(s).repeat(predictions.shape[0],3,s,1).unsqueeze(-1).to(predictions.device))

    #CALCULATE X,Y WIDTH AND HEIGHT WITH SCALING
    x = 1 / s*(box_predictions[...,0:1]+ cell_indices)
    y = 1 / s*(box_predictions[...,1:2]+
               cell_indices.permute(0,1,3,2,4))
    width_height = 1/s*box_predictions[...,2:4]

    #CONCATENATING VALUES AND RESHAPE THEM IN batch_size, num_anchors *s *s, 6
    converted_bboxes = torch.cat((best_class, scores, x, y, width_height), dim = -1).reshape(batch_size, num_anchors *s *s, 6)
    return converted_bboxes.tolist()

def plotting_image_boxes(image, boxes):
    colour_map = plt.get_cmap("tab20b")
    colors = [colour_map(i) for i in np.linspace(0,1,len(class_labels))]

    img = np.array(image)

    h,w = img.shape
    fig,ax = plt.subplots(1)
    ax.imshow(img)

    for box in boxes:

        class_pred = box[0]

        box = box[2:]

        upper_left_x = box[0]-box[2] / 2
        upper_left_y = box[1]-box[3] / 2

        rect = patches.Rectangle((upper_left_x*w, upper_left_y*h),
            box[2]*w,
            box[3],
            linewidth = 2,
            edgecolor = colors[int(class_pred)]
        )
        ax.add_patch(rect)

        plt.text(
            upper_left_x*w,
            upper_left_y*h,
            s=class_labels[int(class_pred)],
            color='white',
            verticalalignment="top",
            bbox={"color":colors[int(class_pred)], "pad":0}
        )
    plt.show()

def save_checkpoint(model,optimizer,filename = "my_checkpoint.pth.tar"):
    print("==> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer" : optimizer.state_dict(),
    }
    torch.save(checkpoint,filename)

def load_checkpoint(checkpoint_file, model, optimizer,lr):
    print("==>Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    



class_labels = [ 
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
device = torch.device("cpu")
load_model = False
save_model = True
checkpoint_file = "checkpoint.pth.tar"

ANCHORS = [[(0.28,0.22),(0.38,0.48),(0.9,0.78)],
            [(0.07,0.15),(0.15,0.11),(0.14,0.29)],
            [(0.02,0.03),(0.04,0.07),(0.08,0.06)]]

batch_size = 32
learning_rate= 1e-5
epochs = 20
image_size = 416

s = [image_size // 32, image_size // 16, image_size // 8]

train_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
        ),
        A.ColorJitter(
            brightness=0.5,contrast=0.5,
            saturation=0.5,hue=0.5,p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean = [0,0,0], std=[1,1,1], max_pixel_value=255
        ),
        transforms.ToTensorV2()
    ],
    bbox_params=A.BboxParams(
        format = "yolo",
        min_visibility=0.4,
        label_fields=[]
    )
)

test_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTRAINT
        ),
        A.Normalize(
            mean = [0,0,0], std = [1,1,1], max_pixel_value=255
        ),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        min_visibility=0.4,
        label_fields=[]
    )
)

dataset = Dataset(
    csv_file = "train.csv",
    image_dir="images/",
    label_dir = "labels/",
    grid_sizes= [13,26,52],
    anchors = ANCHORS,
    transform=test_transform
)

loader = torch.utils.data.DataLoader(
    dataset = dataset,
    batch_size=1,
    shuffle = True
)

GRID_SIZE = [13,26,52]
scaled_anchors = torch.tensor(ANCHORS)/(1/torch.tensor(GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1,3,2))

x,y = next(iter(loader))

boxes = []
for i in range(y[0].shape[1]):
    anchor = scaled_anchors[i]
    boxes += convert_cell_to_bboxes(
        y[i], is_predictions=False, s=y[i].shape[2], anchors = anchor)[0]

boxes = non_maximum_supression(boxes,iou_threshhold=1, threshold=0.7)
plotting_image_boxes(x[0].permute(1,2,0).to("cpu"),boxes)

