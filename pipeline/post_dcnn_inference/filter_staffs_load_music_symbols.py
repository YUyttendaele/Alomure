import pickle5 as pickle
import glob
import os
import numpy as np
import sys
sys.path.append('./data_creation/')
from get_label_dict import get_dict_of_label_ids

label_dict= get_dict_of_label_ids()
inv_label_dict = {v: k for k, v in label_dict.items()}

beginning_labels = ['flat','imaj', 'pmaj', 'imin', 'pmin',\
                                            'imincut', 'pmincut', '3', '2', '1','c clef', 'f clef', 'g clef']
beginning_ids  = [label_dict[label] for label in beginning_labels]
ligature_labels = [ 'l1longa', 'l2longa', 'l1colored longa', 'l2colored longa','l1breve',
                    'l2breve', 'l1colored breve','l2colored breve', 'l1semibreve', 'l2semibreve',
                   'l1colored semibreve', 'l2colored semibreve', 'o1longa','o1colored longa', 'o2longa',             
                    'o2colored longa', 'o1breve', 'o1colored breve', 'o2breve', 'o2colored breve', 
                   'o1semibreve', 'o1colored semibreve', 'o2semibreve', 'o2colored semibreve',
                   'l1','l2','o1','o2','colored l1', 'colored o1', 'colored l2', 'colored o2']
ligatures_ids = [label_dict[label] for label in ligature_labels]  
end_labels = ['custos', 'barline', 'fermata']
end_ids = [label_dict[label] for label in end_labels]  

class Staff_Filtering_and_Music_Symbol_Loading(object):
    def __init__(self,input_folder,img_path,specialized_detections,scale_factor_width,scale_factor_height):   
        self.input_folder= input_folder   
        self.img_path= img_path
        self.specialized_detections= specialized_detections
        self.scale_factor_width = scale_factor_width
        self.scale_factor_height = scale_factor_height
        
    def get_filtered_staffs_with_music_symbols(self):
        staff_dicts,staff_ymins,staff_heights = self.load_staff_dicts() 
        list_of_dictionaries = self.vertically_sort_dictionaries_add_margins_and_staff_heights(staff_dicts,\
                                                                                    staff_ymins,staff_heights)
        list_of_dictionaries = self.filter_false_positive_dictionaries(list_of_dictionaries)    
        list_of_dictionaries = self.rescale_and_round_dictionaries(list_of_dictionaries,self.scale_factor_width,\
                                                                   self.scale_factor_height)
        return list_of_dictionaries
    
    def load_text_model_dict(self):
        text_model_dict = None
        text_model_dict_folder = os.path.join(self.input_folder,'detection_dicts',\
                                         self.img_path.split('/')[-1].split('.jp')[0],'text')
        text_model_dict_path = glob.glob(text_model_dict_folder+'/**.pickle')
        if len(text_model_dict_path) != 0:
            with open(text_model_dict_path[0], 'rb') as handle:
                text_model_dict = pickle.load(handle)
            
        return text_model_dict
    
    
    def text_boxes_within_margins(self,text_model_dict,upper_margin, lower_margin):
        text_boxes_within_staff_margins = []
        score_threshold = 0.2
        boxes = text_model_dict['boxes']
        scores = text_model_dict['scores']
        label_ids = text_model_dict['label_ids']
        for box,score,label_id in zip(boxes,scores,label_ids):
            if label_id == label_dict['text'] and score > score_threshold and\
                box[2] > upper_margin and box[0] < lower_margin:              
                text_boxes_within_staff_margins.append(box)                
        xmins = [box[1] for box in text_boxes_within_staff_margins]
        sorted_idxs = np.argsort(xmins)  
        text_boxes_within_staff_margins= np.asarray(text_boxes_within_staff_margins)[sorted_idxs]                 
        return text_boxes_within_staff_margins
    
    def load_staff_dicts(self):
        staff_dict_folder = os.path.join(self.input_folder,'detection_dicts',\
                                         self.img_path.split('/')[-1].split('.jp')[0])
        staff_dict_paths = glob.glob(staff_dict_folder+'/**.pickle')
        text_model_dict = self.load_text_model_dict()
        staff_dicts= []
        staff_raw_starts= []
        staff_raw_ends= []
        staff_ymins = []
        staff_heights = []        
        left_margin, right_margin,ornate_elements_boxes_and_scores = None, None, [[None],[None]]    
        for counter,dict_path in enumerate(staff_dict_paths): 
            with open(dict_path, 'rb') as handle:
                staff_dict = pickle.load(handle)
            if staff_dict != None and  not 'ornate_elements' in dict_path:    
                staff_ymin = int(dict_path.split('/')[-1].split('_')[0])
                staff_height= int(dict_path.split('_')[-1].split('.pickle')[0])                
                staff_dict['image path'] = self.img_path
                if self.specialized_detections: 
                    for mode in ['beginning','end','ligatures']:#'rare'
                        specialized_detection_dictionary_path = os.path.join(staff_dict_folder,mode,\
                                                        str(staff_ymin)+'_'+str(staff_height)+'.pickle') 
                        if os.path.exists(specialized_detection_dictionary_path):
                            with open(specialized_detection_dictionary_path, 'rb') as handle:
                                specialized_staff_dict = pickle.load(handle)  
                                if specialized_staff_dict != None: 
                                    specialized_staff_dict=\
                                    self.suppress_non_specialized_symbols(specialized_staff_dict,mode)                                                  
                                    staff_dict = self.merge_dict(staff_dict,specialized_staff_dict)
                staff_dict['text boxes'] =[]
                if text_model_dict != None:
                    staff_dict['text boxes'] =self.text_boxes_within_margins(text_model_dict,staff_ymin,\
                                                                             staff_ymin+staff_height)                                                                        
                staff_dict= self.sort_boxes_left_to_right(staff_dict)
                staff_dict['scores'] = np.asarray(staff_dict['scores'],dtype=float)
                staff_dict['boxes'] = np.asarray(staff_dict['boxes'],dtype=int)
                staff_dict['text boxes'] = np.asarray(staff_dict['text boxes'],dtype=int)
                staff_dict['label_ids'] = np.asarray(staff_dict['label_ids'],dtype=int)                                
                staff_dicts.append(staff_dict)
                staff_ymins.append(staff_ymin)
                staff_heights.append(staff_height)  
          
        return staff_dicts,staff_ymins,staff_heights      
    
    def vertically_sort_dictionaries_add_margins_and_staff_heights(self,staff_dicts,staff_ymins,staff_heights):       
        sorted_indices= np.argsort(staff_ymins)
        staff_dicts = np.asarray(staff_dicts)[sorted_indices]
        staff_ymins = np.asarray(staff_ymins)[sorted_indices]
        staff_heights = np.asarray(staff_heights)[sorted_indices]
       
        likely_staff_height = np.mean(staff_heights)
         #regularize staff height
        for i in range(0,len(staff_heights)):
            if staff_heights[i] < likely_staff_height - int(likely_staff_height/5):
                dif = np.abs(likely_staff_height-staff_heights[i])
                staff_heights[i] = staff_heights[i]+ dif
                staff_ymins[i] = staff_ymins[i]-int(dif/2)  
                
        #get right margins
        right_margins = []
        for staff_dict in staff_dicts:
            right_margin = staff_dict['boxes'][-1][-1]
            right_margins.append(right_margin) 
        right_margin = np.median(right_margins)
        #distill music symbol dictionaries
        list_of_dictionaries = []      
        for staffdict,staff_height,staff_ymin in zip(staff_dicts,staff_heights,staff_ymins): 
            staffdict['ymin'] = staff_ymin
            staffdict['staff_height'] = staff_height
            staffdict['right margin'] = right_margin
            list_of_dictionaries.append(staffdict)
        return list_of_dictionaries      

    def filter_false_positive_dictionaries(self,list_of_dictionaries):    
        kept_dictionaries = []
        inter_staff_spacings = []
        all_scores = []
        staff_dict_scores=[]
        for i,staff_dict in enumerate(list_of_dictionaries):
            all_scores+=list(staff_dict['scores'])
            staff_dict_scores.append(staff_dict['scores'])       
        # sanity check
        for i,dict in enumerate(list_of_dictionaries):
            if len(dict['boxes']) >2 and np.mean(staff_dict_scores[i]) > np.mean(all_scores)- 2*np.std(all_scores):
                kept_dictionaries.append(dict)
        return kept_dictionaries 

    def rescale_and_round_dictionaries(self,list_of_dict,scale_factor_width,scale_factor_height):
        rescaled_list_of_dictionaries = []
        if scale_factor_width==None and scale_factor_height==None:
            scale_factor_width = 1
            scale_factor_height = 1
        for i in range(len(list_of_dict)):
            dict={}
            dict['image path'] = list_of_dict[i]['image path']
            rescaled_boxes = []
            for box in list_of_dict[i]['boxes']:
                box = np.asarray(box)
                rescaled_box = np.asarray(np.round(box*[scale_factor_height,scale_factor_width,\
                                                        scale_factor_height,scale_factor_width]),dtype=int)
                rescaled_boxes.append(rescaled_box)
                
            rescaled_text_boxes = []    
            for box in list_of_dict[i]['text boxes']:
                box = np.asarray(box)
                rescaled_box = np.asarray(np.round(box*[scale_factor_height,scale_factor_width,\
                                                        scale_factor_height,scale_factor_width]),dtype=int)
                rescaled_text_boxes.append(rescaled_box)                
            dict['boxes'] =rescaled_boxes
            dict['text boxes'] =rescaled_text_boxes
            dict['right margin'] =list_of_dict[i]['right margin']
            
            dict['label_ids'] = list_of_dict[i]['label_ids']
            dict['scores'] = list_of_dict[i]['scores']
            dict['ymin'] = int(np.round(scale_factor_height*list_of_dict[i]['ymin']))
            dict['staff_height'] =int(np.round(scale_factor_height*list_of_dict[i]['staff_height']))
            rescaled_list_of_dictionaries.append(dict)   
        return rescaled_list_of_dictionaries         
    
    def merge_dict(self,dict1, dict2):    
        dict1['scores']= np.append(dict1['scores'],dict2['scores'],axis=0)
        dict1['label_ids']= np.append(dict1['label_ids'],dict2['label_ids'],axis=0)    
        dict1['boxes']= np.append(dict1['boxes'], dict2['boxes'] ,axis=0)   
        return dict1           
    
    def sort_boxes_left_to_right(self, staff_dict): 
        xmins = [box[1] for box in staff_dict["boxes"]]
        sorted_idxs = np.argsort(xmins)  
        for key in ['scores','label_ids','boxes']:
            staff_dict[key]= np.asarray(staff_dict[key])[sorted_idxs]
        return staff_dict   

    def suppress_non_specialized_symbols(self,specialized_dict,mode):
        if mode == 'beginning':
            specialized_ids = beginning_ids 
        elif mode == 'end':
            specialized_ids= end_ids
        elif mode == 'ligatures':
            specialized_ids = ligatures_ids
        else: 
            specialized_ids = label_dict.values()
        kept_indices = [index for index in range(len(specialized_dict['label_ids']))\
                       if specialized_dict['label_ids'][index] in specialized_ids]
        for key in ['label_ids','scores','boxes']:
            specialized_dict[key] = np.asarray(specialized_dict[key])[kept_indices]               
        return specialized_dict 
