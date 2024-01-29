import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

class rest_nn(nn.Module):
    def __init__(self):
        """Initialize the model from the predefined weights path"""
        super(rest_nn, self).__init__()
        self.model_save_path = './model_weights/rest_classification.pth'
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

            output = output+ hor_output
            pred_label_id = output.data.max(1, keepdim=True)[1].item()
            predicted_label = self.inv_label_dict[pred_label_id]
        return predicted_label    
