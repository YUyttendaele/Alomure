import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import  models

class staff_position_nn(nn.Module):
    """
    Contains model definitions and methods for inferring staff positions.
    """
    def __init__(self):
        super(staff_position_nn, self).__init__()
        
        self.resize_dim = 224       
        self.n_possible_pos = 15
        self.ledger_pos_dict = {0:0,1:1,2:2,3:12,4:13,5:14}
        self.clef_pos_dict = {0:3,1:5,2:7,3:9,4:11}
        
        self.general_model_save_path = './model_weights/mensural_pitch_classification.pth'
        self.ledger_model_save_path = './model_weights/mensural_pitch_classification_ledger.pth' 
        self.clef_model_save_path = './model_weights/mensural_pitch_classification_clef.pth' 
        
        self.general_model,self.device =self.load_staff_position_model(self.general_model_save_path,
                                                                       self.n_possible_pos)
        self.ledger_model,_ =self.load_staff_position_model(self.ledger_model_save_path,
                                                                      len(self.ledger_pos_dict))
        self.clef_model,_ =self.load_staff_position_model(self.clef_model_save_path,
                                                                      len(self.clef_pos_dict))    
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

    def load_staff_position_model(self,model_save_path,n_possible_pos):        
        resnet_18=  models.resnet18(pretrained=False)
        num_ftrs = resnet_18.fc.in_features
        resnet_18.fc = nn.Linear(num_ftrs, n_possible_pos)
        checkpoint = torch.load(model_save_path )
        resnet_18.load_state_dict(checkpoint['model_state_dict'])     
        
        device = self.get_default_device()
        resnet_18= self.to_device(resnet_18, device)
        return resnet_18,device

    def predict_staff_position(self,img):
        with torch.no_grad():
            self.general_model.eval()

            img = torch.tensor(img, dtype = torch.float32).permute(2,0,1)
            or_img=transforms.Resize(size=(self.resize_dim,self.resize_dim))(img) 
            img=torch.unsqueeze(or_img, dim=0)  
            img= self.to_device(img, self.device)
            output = self.general_model(img)

            hor_image = transforms.functional.hflip(or_img)
            hor_image =torch.unsqueeze(hor_image, dim=0)
            hor_image = self.to_device(hor_image, self.device)
            hor_output = self.general_model(hor_image)

            ver_image = transforms.functional.vflip(or_img)
            ver_image =torch.unsqueeze(ver_image, dim=0)
            ver_image = self.to_device(ver_image, self.device)
            ver_output = self.general_model(ver_image)

            output_sum = output+ hor_output+ torch.flip(ver_output,dims=(1,))
            total_vote = output_sum.argmax().item()     

            if total_vote in self.ledger_pos_dict.values():
                total_vote = self.predict_ledger_position(or_img)                
            return total_vote
        
    def predict_ledger_position(self,or_img):
        with torch.no_grad():
            self.ledger_model.eval()
            
            img=torch.unsqueeze(or_img, dim=0)  
            img= self.to_device(img, self.device)        
            output = self.ledger_model(img)

            hor_image = transforms.functional.hflip(or_img)
            hor_image =torch.unsqueeze(hor_image, dim=0)
            hor_image = self.to_device(hor_image, self.device)
            hor_output = self.ledger_model(hor_image)

            ver_image = transforms.functional.vflip(or_img)
            ver_image =torch.unsqueeze(ver_image, dim=0)
            ver_image = self.to_device(ver_image, self.device)
            ver_output = self.ledger_model(ver_image)

            output_sum = output+ hor_output+ torch.flip(ver_output,dims=(1,))
            total_vote = output_sum.argmax().item()     
            return self.ledger_pos_dict[total_vote]        
               
    def predict_clef_position(self,img):
        with torch.no_grad():
            self.clef_model.eval()

            img = torch.tensor(img, dtype = torch.float32).permute(2,0,1)
            img=transforms.Resize(size=(self.resize_dim,self.resize_dim))(img) 
            img=torch.unsqueeze(img, dim=0)  
            img= self.to_device(img, self.device)
            
            output = self.clef_model(img)
            pred = output.data.max(1, keepdim=True)[1].item()
            
            return self.clef_pos_dict[pred]
