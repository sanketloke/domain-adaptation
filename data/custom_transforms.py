class ToLabelTensor(object):
    """Takes a tensor image as input. 
    Linearly scans over each pixel
    converts it into label tensor according to the labels provided
    """

    # TODO: Very inefficient implementation threading might alleviate the issue. 
    def image2label(self,tensorImage):
        
        import torchvision
        import torch
        import numpy as np
        to_pil = torchvision.transforms.ToPILImage()
        img= to_pil(tensorImage)
        from scipy import misc
        image_array=np.ascontiguousarray(misc.fromimage(img))
        labels= self.labels
        id2color={ label.id : label.color[2]*pow(10,6) + label.color[1]*pow(10,3) + label.color[0] for label in reversed(labels) }
        color2id = dict(reversed(item) for item in id2color.items())
        id2trainId ={ label.id : label.trainId for label in reversed(labels) }
        shape= (image_array.shape[0],image_array.shape[1])
        label_map = torch.zeros(shape[0],shape[1])
        mod_arr=np.zeros((image_array.shape[0],image_array.shape[1]))
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                mod_arr[i][j]=image_array[i][j][2]*pow(10,6) + image_array[i][j][1]*pow(10,3) + image_array[i][j][0]
                if mod_arr[i][j] not in color2id:
                    mod_arr[i][j]= 0
        def check_val(temp):
            class_val = id2trainId[color2id[temp]]
            if int(class_val)== 225 or class_val is 255:
                class_val=20
            if int(class_val)== -1:
                class_val=19
            return class_val
        c=np.vectorize(check_val)
        mod_arr=c(mod_arr)
        #print torch.from_numpy(mod_arr)
        return torch.from_numpy(mod_arr)

    def label2image(self,tensorImage):
        import torchvision
        import torch
        import numpy as np
        labels= self.labels
        id2color={ label.id : (label.color[2],label.color[1], label.color[0]) for label in reversed(labels) }
        color2id = dict(reversed(item) for item in id2color.items())
        id2trainId ={ label.id : label.trainId for label in reversed(labels) }
        id2colorval={ label.id : ( label.color[0],label.color[1],label.color[2]) for label in reversed(labels) }
        trainId2id=dict(reversed(item) for item in id2trainId.items())
        trainId2id[19]=-1
        trainId2id[20]=30
        def reverse_val(temp):
            if int(temp)==20:
                temp=-1
            if int(temp)==21:
                temp=255
            class_val = id2colorval[trainId2id[temp]]
            return class_val
        z=np.vectorize(reverse_val)
        
        a=np.dstack(z(tensorImage))
        return a.astype(np.uint8)



    def __init__(self, labels):
        self.labels = labels

    def __call__(self, img):
        return self.image2label(img)
    

import torch
from pdb import set_trace as st
class DownSizeLabelTensor(object):

    def downsize(self,img):
        i=img.size()[0]
        j=img.size()[1]
        q= torch.nn.AvgPool2d(self.factor)(torch.autograd.Variable(img).float()).int()
        return q

    def findDecreasedResolution(self,size):
        return torch.nn.AvgPool2d(self.factor)(torch.autograd.Variable(torch.randn(1,size,size)).float()).int().size()[1]

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        a=[]
        # print 'Called!'
        # print img
        # print img.expand(1,img.size()[0],img.size()[1])
        # print self.downsize(img.expand(1,img.size()[0],img.size()[1]))
        return self.downsize(img.expand(1,img.size()[0],img.size()[1])).data
