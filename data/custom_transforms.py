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
        #print image_array.T
        #return image_array
        labels= self.labels
        id2color={ label.id : label.color[0]*pow(10,6) + label.color[1]*pow(10,3) + label.color[0] for label in reversed(labels) }
        color2id = dict(reversed(item) for item in id2color.items())
        id2trainId ={ label.id : label.trainId for label in reversed(labels) }
        shape= (image_array.shape[0],image_array.shape[1])
        label_map = torch.zeros(shape[0],shape[1])
        mod_arr=np.zeros((image_array.shape[0],image_array.shape[1]))
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                mod_arr[i][j]=image_array[i][j][0]*pow(10,6) + image_array[i][j][1]*pow(10,3) + image_array[i][j][0]
        #print mod_arr  
        def check_val(temp):
            class_val = id2trainId[color2id[temp]]
            if int(class_val)== 225 or class_val is 255:
                class_val=21
            if int(class_val)== -1:
                class_val=20
            return class_val
        c=np.vectorize(check_val)
        mod_arr=c(mod_arr)
        return torch.from_numpy(mod_arr)
#         for i in range(0,shape[0]):
#             for j in range(0,shape[1]):
#                 temp=image_array[i][j][0]*pow(10,6) + image_array[i][j][1]*pow(10,3) + image_array[i][j][0]
#                 try:
#                     if temp in color2id:
#                         class_val = id2trainId[color2id[temp]]
#                         if class_val is 225:
#                             class_val=19
#                         if class_val is -1:
#                             class_val=20
#                         label_map[i][j] = class_val
#                     else:
#                         print 'error'
#                         print tuple(image_array[j,i])
#                         break
#                 except:
#                     print 'error'
#                     print i,j
#                     print tuple(image_array[i,j])
#                     return
#         return label_map


    def __init__(self, labels):
        self.labels = labels
        print 'Initialized'

    def __call__(self, img):
        return self.image2label(img)
