import cv2
import torch
import torch.nn as nn
from torchvision import models

class Siamese_Net_Structure(nn.Module):
 
    """ Defines the siamese neural network structure given the classifier model"""

    def __init__(self, classification_model):
        super().__init__()

        self.network = classification_model
        self.last_fc =  nn.Linear(1,1)
              
        self.distance = nn.PairwiseDistance(p=1)
        self.bn1d  = nn.BatchNorm1d(1)       
        
    def forward_once(self, x):    
        x= self.network(x)
        x= nn.Flatten()(x)
        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)        
        output2 = self.forward_once(input2)
        
        dist= self.distance(output1, output2)
        dist = torch.unsqueeze(dist, dim=1)
        dist = self.bn1d(dist)
        dist=self.last_fc(dist)        
        
        similarity_score= nn.Sigmoid()(dist) 
        return similarity_score.reshape(-1) 
    
class Siamese_inference():
    
    """ Estimates the similarity between two images, maximum similarity := 1 , minimum similarity := 0 """
    
    def __init__(self):
        super(Siamese_inference, self).__init__()
            
        #base classifier
        NUM_OUT_FTRS = 16
        resnet_18 = models.resnet18(pretrained=False)
        resnet_18.fc = nn.Linear(resnet_18.fc.in_features, NUM_OUT_FTRS )     

        #load siamese network
        model = Siamese_Net_Structure(classification_model = resnet_18)
        model_save_path = './model_weights/mensural_siam_resnet'
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        #load to device
        self.device= self.get_default_device()
        self.model = self.to_device(model, self.device)
        
        #input image size
        self.resize_dim = 224
             
    def get_default_device(self):
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def to_device(self,data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True) 
    
    def transform_input_img(self,img):
        img = cv2.resize(img, (self.resize_dim, self.resize_dim))
        img = torch.tensor(img, dtype = torch.float32).permute(2,0,1)
        img = torch.unsqueeze(img, dim=0)
        img = self.to_device(img, self.device)
        return img
    
    def get_similarity_score(self,img0,img1):
        img0 = self.transform_input_img(img0)
        img1 = self.transform_input_img(img1)
        
        with torch.no_grad():
            self.model.eval()
            pred = self.model(img0, img1).item()
        return pred      
    
