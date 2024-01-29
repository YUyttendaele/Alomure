import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import sys
sys.path.append('../data_creation/')
from get_label_dict import get_dict_of_label_ids
label_dict= get_dict_of_label_ids()
inv_label_dict = {v: k for k, v in label_dict.items()}

class classification_nn(nn.Module):
    def __init__(self):
        """Initialize the model from the predefined weights path"""
        super(classification_nn, self).__init__()
        self.model_save_path = './model_weights/mensural_resnet_18_classification.pth'
        self.resize_dim = 224
        self.model,self.device,self.label_dict =self.load_classification_model()
        self.inv_label_dict =  {v: k for k, v in self.label_dict.items()}

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

    def load_classification_model(self): 
        """Load ResNet18 model"""
        device = self.get_default_device()
        checkpoint = torch.load(self.model_save_path)
        classes_dict  = checkpoint['label_dict']
        resnet_18=  models.resnet18(pretrained=False)
        num_ftrs = resnet_18.fc.in_features
        resnet_18.fc = nn.Linear(num_ftrs, len(classes_dict.keys()))
        resnet_18.load_state_dict(checkpoint['model_state_dict'])               
        resnet_18= self.to_device(resnet_18, device)
        return resnet_18,device,classes_dict
    
    def load_image(self,img):
        """Tensorize and transform images in accordance with ResNet requirements"""           
        img = torch.tensor(img, dtype = torch.float32).permute(2,0,1)
        img=transforms.Resize(size=(self.resize_dim,self.resize_dim))(img) 
        img=torch.unsqueeze(img, dim=0)  
        img= self.to_device(img, self.device)  
        return img
    
    
    def predict_label(self,img):
        """Predict labels"""
        with torch.no_grad():
            self.model.eval()
            img = torch.tensor(img, dtype = torch.float32).permute(2,0,1)
            or_img=transforms.Resize(size=(self.resize_dim,self.resize_dim))(img) 
            img=torch.unsqueeze(or_img, dim=0)  
            img= self.to_device(img, self.device)      
            output = self.model(img)

            hor_image = transforms.functional.hflip(or_img)
            hor_image =torch.unsqueeze(hor_image, dim=0)
            hor_image = self.to_device(hor_image, self.device)
            hor_output = self.model(hor_image)

            ver_image = transforms.functional.vflip(or_img)
            ver_image =torch.unsqueeze(ver_image, dim=0)
            ver_image = self.to_device(ver_image, self.device)
            ver_output = self.model(ver_image)

            output = output+ hor_output+ ver_output
            _,indices = torch.topk(output.data,5)
        return [self.inv_label_dict[index] for index in indices.cpu().numpy()[0]]
