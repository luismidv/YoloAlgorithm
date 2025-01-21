import torch 
import torch.nn as nn
import numpy as np

class Yololayer(nn.Module):
    def __init__(self,anchor_mask=[], num_classes = 0, anchors = [], num_anchors = 1):
        super(Yololayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = 32
        self. seen = 0
    
    def forward(self,output,nms_tresh):
        self.thresh = nms_tresh
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m*self.anchor_step:(m+1)*self.anchor_step]

        masked_anchors = [self.anchors/self.stride for anchor in masked_anchors]
        boxes = get_region_boxes(output.data,self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask))

        return boxes
    
class Upsample(nn.Module):
    def __init__(self,stride = 2):
        super(Upsample, self).__init__()
        self.stride = stride
    def forward(self,x):
        stride = self.stride
        assert(x.data.dim()==4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B,C,H,1,W,1).expand(B,C,H,stride,W,stride).contiguous().view(B,C,H*stride, W*stride)
        return x
    
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self,x):
        return x
    
class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks)
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def forward(self, x, nms_thresh):
        ind = -2
        self.loss = None
        output = dict()
        out_boxes = []

        for block in self.blocks:
            ind = ind + 1
            if block["type"] == 'net':
                continue
            elif block['type'] in ['convolutional', 'upsample']:
                x = self.models[ind](x)
                output[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]

                if len(layers) == 1:
                    x = output[layers[0]]
                    output[ind] = x
                elif len(layers) == 2:
                    x1 = output[layers[0]]
                    x2 = output[layers[1]]
                    x = torch.cat((x1,x2),1)
                    output[ind] = x
            elif block['type'] == 'shorcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = output[from_layer]
                x2 = output[ind-1]
                x = x1 + x2
            elif block['type'] == 'yolo':
                boxes = self.models[ind](x,nms_thresh)
                out_boxes.append(boxes)
            else:
                print('unknown type %s' % (block['type']))
        return out_boxes
    
    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self,blocks):
        models = nn.ModuleList()

        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0

        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)//2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))

                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
            
            elif block["type"] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride= prev_stride//stride
                out_strides.append(prev_stride)
                models.append(Upsample(stride))

            elif block["type"] == 'route':
                layers = block['layers.split']
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind(i) + ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_stride = out_strides[layers[0]]
                    out_strides.append(prev_stride)
                    models.append(EmptyModule())

            elif block["type"] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())

            elif block['type'] == 'yolo':
                yolo_layer = Yololayer()
                anchor = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors)//yolo_layer.num_anchors
                yolo_layer.stride = prev_stride
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)

            else:
                print('Uknown type %s' %(block['type']))
        return models
    
    def load_weights(self,weightfile):
        print()
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count = 5, dtype = np.int32)
        self.header = torch.from_numpy(header)
        self.seen=self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        counter = 3
        
        for block in self.blocks:
            if start >= buf.size:
                break
            
            ind = ind + 1
            if block['type'] == 'net':
                continue
            
            elif block['type'] == 'convolutional':
                model = self.model[ind]
                batch_normalize = int(block['batch_normalize'])
                #CHECK THIS
                if batch_normalize:
                    start = load_conv_bn(buf,start,model[0], model[1])
                else:
                    start = load_conv(buf,start, model[0])

            elif block['type'] == 'upsample':
                pass
            
            elif block['type'] == 'route':
                pass
            
            elif block['tpye'] == 'shortcut':
                pass
            
            elif block['type'] == 'yolo':
                pass
            
            else:
                print("Unknown type %s" %(block['type']))

            percent_comp = (counter/len(self.blocks['type']))*100

            print('Loading wight. Please wait...{:.2f}% Complete'.format(percent_comp), end = '\r', flush = True)
            
            counter += 1
    
    def convert2cpu(gpu_matrix):
        return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)
    
    def convert2cpu_long(gpu_matrix):
        return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)
    
    def get_region_boxes(output, conf_thresh, num_classes, anchors , num_anchors
                         ,only_objectness = 1, validaiton = False):
        anchor_step = len(anchors)// num_anchors
        if output.dim() == 3:
            output = output.unsqueeze(0)
        batch = output.size(0)
        assert(output.size(1) == (5+ num_classes)*num_anchors)
        h = output.size(2)
        w = output.size(3)

        all_boxes = []
        output = output.view(batch*num_anchors, 5 + num_classes, h * w).tranpose(0,1).contiguous().view(5 + num_classes)

        #CREATE 1 DIMENSION TENSOR
        grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).type_as(output)
        grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors,1,1).view(batch*num_anchors*h*w).type_as(output)
        
        xs = torch.sigmoid(output[0]) + grid_x
        ys = torch.sigmoid(output[0]) + grid_y

        anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1,torch.LongTensor([0]))
        anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1,torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(batch,1).repeat(1,1,h*w).view(batch*num_anchors*h*w).type_as(output)
        anchor_h = anchor_h.repeat(batch,1).repeat(1,1,h*w).view(batch*num_anchors*h*w).type_as(output)

        ws = torch.exp(output[2]) * anchor_w
        hs = torch.exp(output[3])*anchor_h

        det_confs = torch.sigmoid(output[4])
        cls_confs = torch.nn.Softmax(dim = 1)[5:5+num_classes]
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)



