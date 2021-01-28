#################################################################
#--------------------------import-------------------------------#
#################################################################
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


#################################################################
#---------------------------Def---------------------------------#
#################################################################


class Iclver_Data(Dataset):    
    def  __init__(self, json_file_objects, json_file_train_image, image_file):
        super(Iclver_Data, self).__init__()
        """
           json_file_objects : label 0~23
           json_file_train_image : the label containing in each training image
           image_file : the image path 
        """
        self.json_file_objects = json_file_objects
        self.json_file_train_image = json_file_train_image
        self.image_file = image_file
        self.transform = transforms.Compose([
                                             transforms.Resize((64,64)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])
        #---read line by line ---
        #read label index (dictionary)
        with open(self.json_file_objects, 'r') as file:
            self.objects_dict = json.load(file)
        #read training image with labels 
        with open(self.json_file_train_image, 'r') as file:
            image_dict = json.load(file)
            self.image_list = [[item[0], item[1]] for item in image_dict.items()]
        
        # image_list -> CLEVR_train_004265_2.png', ['purple cylinder', 'blue cube', 'purple sphere']....
        print("> Found %d images... successfully" % (len(self.image_list)))

            
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        """
           image_list[index][0] : CLEVR_train_002066_0.png
           image_list[index][1] : ['cyan cube']
        """
        image_name, image_conditions = self.image_list[index][0], self.image_list[index][1]
        image = Image.open(self.image_file + image_name).convert('RGB')
        image = self.transform(image)
        label = torch.zeros(24)
        for idx, object in enumerate(image_conditions):
            label[self.objects_dict[object]] = 1
        # print(image.shape,label.shape) -> torch.Size([3, 64, 64]) torch.Size([24])
        return image, label
    