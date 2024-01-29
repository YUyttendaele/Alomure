ligature_notes= ['l1longa', 'l2longa', 'l1colored longa', 'l2colored longa', 'l1breve', 'l2breve',
                       'l1colored breve', 'l2colored breve', 'l1semibreve', 'l2semibreve', 'l1colored semibreve',\
                       'l2colored semibreve', 'o1longa', 'o1colored longa', 'o2longa', 'o2colored longa', 'o1breve',\
                       'o1colored breve', 'o2breve', 'o2colored breve', 'o1semibreve', 'o1colored semibreve', 'o2semibreve',\
                       'o2colored semibreve','l1', 'l2', 'o1', 'o2', 'colored l1', 'colored o1', 'colored l2', 'colored o2']
import numpy as np
import os
import cv2
import glob
import time
import pickle
import sys
import shutil
from tqdm import tqdm
from pathlib import Path
from cv2 import dnn_superres

sys.path.append('./dcnns/yolov7/')
sys.path.append('./post_dcnn_inference/')
sys.path.append('./data_creation/')
from detect import *
from get_label_dict import get_dict_of_label_ids
label_dict = get_dict_of_label_ids()
inv_label_dict = {v: k for k, v in label_dict.items()}

ligature_ids= [label_dict[label] for label in ligature_notes]

class object_detection_yolov7(object):
    
    def __init__(self,input_folder,aspect_ratio):        
        self.input_folder = input_folder
        self.resize_dim = 640
        self.slice_overlap_fraction = 0.25
        self.aspect_ratio= aspect_ratio
        self.detection_dicts_dir =os.path.join(self.input_folder,'detection_dicts')  
        self.staff_inference_dir  = os.path.join(self.input_folder,'inference', 'staffs')      
        self.non_rescaled_images_dir  = os.path.join(self.input_folder,'non_rescaled_images')      

        self.text_image_slices_dir = os.path.join(self.input_folder,'sliced_text_images')
        self.text_slices_inference_dir =  os.path.join(self.input_folder,'inference','text')         
        
        self.sliced_staff_images_dir= os.path.join(self.input_folder, 'sliced_staff_images') 
        self.sliced_staff_inference_dir_all= os.path.join(self.input_folder,'inference', 'slices_all')
        self.sliced_staff_inference_dir_rare= os.path.join(self.input_folder,'inference', 'slices_rare')
           
        
        self.ligature_image_slices_dir = os.path.join(self.input_folder,'sliced_staff_images','ligatures')
        self.ligature_slices_inference_dir =  os.path.join(self.input_folder,'inference','ligatures')     
      
        self.begin_image_slices_dir = os.path.join(self.input_folder,'sliced_staff_images','beginning')
        self.begin_slices_inference_dir =  os.path.join(self.input_folder,'inference','beginning')   
        
        self.end_image_slices_dir = os.path.join(self.input_folder,'sliced_staff_images','end')
        self.end_slices_inference_dir =  os.path.join(self.input_folder,'inference','end')   

   
        #create output folders      
        for directory in [self.detection_dicts_dir, self.staff_inference_dir,self.text_slices_inference_dir,\
                          self.text_image_slices_dir, self.ligature_image_slices_dir,\
                          self.ligature_slices_inference_dir,self.sliced_staff_images_dir,\
                          self.sliced_staff_inference_dir_all,self.begin_image_slices_dir,\
                          self.begin_slices_inference_dir, self.end_image_slices_dir,\
                          self.end_slices_inference_dir,self.non_rescaled_images_dir,\
                          self.sliced_staff_inference_dir_rare]:
            
             if not(os.path.exists(directory)):            
                os.makedirs(directory) 
    def inference(self,model_type, image_path,output_folder):
        if model_type == 'staff':
            weights = './model_weights/yolo_model_weights/staff/staff.pt'

            img_size = str(1024)
            conf = str(0.3)
            iou = str(0.2)
        elif model_type == 'text':
            weights = './model_weights/yolo_model_weights/text/best.pt'           
            img_size = str(1024)
            conf = str(0.2)
            iou = str(0.2)
            
            
            
        elif model_type == 'beginning':
            weights = './model_weights/yolo_model_weights/begin/best.pt'           
            img_size = str(800)
            conf = str(0.15)
            iou = str(0.45)            
        elif model_type == 'end':
            
            weights = './model_weights/yolo_model_weights/end/best.pt'
            
            img_size = str(640)
            conf = str(0.15)
            iou = str(0.45)                
        elif model_type == 'ligature':
            weights = './model_weights/yolo_model_weights/ligatures/best.pt'
            img_size = str(640)
            conf = str(0.2)
            iou = str(0.7)   
            
        elif model_type == 'rare': # 
            weights = './model_weights/yolo_model_weights/rare/best.pt'
            img_size = str(640)    
            conf = str(0.15)
            iou = str(0.45)
            
            
        else: #model_type == 'all music symbols'
            weights = './model_weights/yolo_model_weights/rare/best.pt'       
            img_size = str(640)    
            conf = str(0.15)
            iou = str(0.45)
        source = image_path
        name =  ''                    
        parser = make_parser()
        opt = parser.parse_args([
            '--weights', weights,
            '--source', source,
            '--save-conf',
            '--save-txt',
            '--img-size', img_size,
            '--project', output_folder,
            '--name', name,
            '--exist-ok',
            '--conf-thres', conf,
            '--iou-thres', iou,
        ])
        with torch.no_grad():
                detect(opt) 
                
    def get_dict(self,file,vertical_flip):
        label_ids = []
        scores  = [] 
        x_center_norms= []
        y_center_norms= []
        width_norms= []
        height_norms= []
        f=open(file,"r")
        lines=f.readlines()
        result=[]
        for x in lines:
            label_ids.append(int(x.split(' ')[0]))
            scores.append(float(x.split(' ')[1]))
            x_center_norms.append(float(x.split(' ')[2]))
            y_center_norms.append(float(x.split(' ')[3]))
            width_norms.append(float(x.split(' ')[4]))
            height_norms.append(float(x.split(' ')[5]))       
        boxes = []
        for (x_center_norm,y_center_norm,width_norm,height_norm) in zip(x_center_norms,y_center_norms,width_norms,height_norms):
            ymin= (y_center_norm -0.5*height_norm)
            xmin = (x_center_norm -0.5*width_norm)
            ymax = (y_center_norm + 0.5*height_norm)
            xmax = (x_center_norm + 0.5*width_norm)
            if vertical_flip: 
                ymax = 1-ymin
                ymin = 1-ymax
            box = [ymin,xmin,ymax,xmax]                                                                        
            boxes.append(np.asarray(box))
        dict = {}    
        dict['scores'] = np.asarray(scores)
        dict['label_ids'] = np.asarray(label_ids)
        dict['boxes'] = np.asarray(boxes)
        return dict     
    
    def merge_dict(self,dict1, dict2):    
        dict1['scores']= np.append(dict1['scores'],dict2['scores'],axis=0)
        dict1['label_ids']= np.append(dict1['label_ids'],dict2['label_ids'],axis=0)    
        dict1['boxes']= np.append(dict1['boxes'], dict2['boxes'] ,axis=0)   
        return dict1                

    def slice_staff_horizontally_into_windows(self,staff_start,staff_end,mean_staff_height,left_margin):
        staff_slice_width = self.aspect_ratio*mean_staff_height
        xcoordinates_start =[staff_start]  
        staff_widths = [staff_slice_width]
        while xcoordinates_start[-1]+ staff_slice_width <staff_end:
            xcoordinates_start.append(xcoordinates_start[-1]+(1-self.slice_overlap_fraction)*staff_slice_width)    
            staff_widths.append(staff_slice_width)
        if staff_end -staff_slice_width >0:
            xcoordinates_start[-1] = np.maximum(left_margin,staff_end -staff_slice_width)

        xcoordinates_start = np.asarray(xcoordinates_start)
        xcoordinates_end=xcoordinates_start+ np.asarray(staff_widths)
        sorted_indices= np.argsort(xcoordinates_end)
        xcoordinates_end = np.asarray(xcoordinates_end,dtype=int)[sorted_indices]          
        xcoordinates_start = np.asarray(xcoordinates_start,dtype=int)[sorted_indices]  
        return xcoordinates_start,xcoordinates_end    

    
    def get_staffs_from_dict(self,staffdict):
        """
        filter non-staff symbols
        """ 
        staff_boxes = []
        for box, label_id in zip(staffdict['boxes'], staffdict['label_ids']):
            if label_id  == label_dict['staff']: 
                staff_boxes.append(box)
        return staff_boxes
    
    def get_filtered_staffs_and_page_margins(self,staffs):       
        # determine margins by filtering outliers
        #filter 
        TRIM_PERCENTAGE = 0.2
        MINIMUM_STAFF_WITHIN_MARGINS_PCT = 0.8
        left_margins = [staff[1] for staff in staffs]
        right_margins = [staff[3] for staff in staffs]
        sorted_left_margins = np.sort(left_margins)
        sorted_right_margins = np.sort(right_margins)
        left_trim_index = int(TRIM_PERCENTAGE*len(staffs))
        right_trimindex= len(staffs) - int(TRIM_PERCENTAGE*len(staffs))
        trimmed_mean_left = np.mean(sorted_left_margins[left_trim_index: right_trimindex])
        trimmed_mean_right = np.mean(sorted_right_margins[left_trim_index: right_trimindex])
        filtered_staffs = []
        for staff in staffs :
            staff_left = staff[1]
            staff_right = staff[3]
            if not( (trimmed_mean_left-staff_left)/(staff_right-staff_left)> MINIMUM_STAFF_WITHIN_MARGINS_PCT  or\
                   (staff_right-trimmed_mean_right)/(staff_right-staff_left)> MINIMUM_STAFF_WITHIN_MARGINS_PCT  ):
                filtered_staffs.append(staff)       
        left_margin =  np.min([staff[1] for staff in filtered_staffs])
        right_margin =  np.max([staff[3] for staff in filtered_staffs])

        return filtered_staffs, left_margin, right_margin
    
    
    def vert_overlap_box(self,box1,box2):
        ymmin_box1 = box1[0]
        ymax_box1 = box1[2]
        ymmin_box2 = box2[0]
        ymax_box2 = box2[2]        
        return (np.minimum(ymax_box2,ymax_box1)-np.maximum(ymmin_box1,ymmin_box2))/(ymax_box2-ymmin_box2 )


    def get_ornate_elem_from_dict(self,staff_dict,left_margin, right_margin):
        ornate_boxes = []
        scores = []
        for box, score, label_id in zip(staff_dict['boxes'],staff_dict['scores'], staff_dict['label_ids']):
            if label_id  == label_dict['ornate element'] and box[3] >left_margin and box[1] < right_margin:
                ornate_boxes.append(box)
                scores.append(score)
        return ornate_boxes,scores
        
    def get_stave_crop_coordinates(self,staff_dict,image_height):
        staffs = self.get_staffs_from_dict(staff_dict)  
        # we assume only 1 page has music symbol input, everything else is false posive
        # and needs to be filtered
        staffs,left_margin, right_margin = self.get_filtered_staffs_and_page_margins(staffs)
        staffs = self.vert_overlap_staff(staffs)
        ornate_elements_boxes,ornate_elements_scores = self.get_ornate_elem_from_dict(staff_dict,left_margin,right_margin)
        
        ycoordinates, xcoordinates, staff_heights = [], [],[]
        for staff in staffs:
            ycoordinates.append([int(staff[0]),int(staff[2])]) 
            xcoordinates.append([int(staff[1]), int(staff[3])] )
            staff_heights.append(-int(staff[0]) + int(staff[2]))  
        
        #adjust y coordinates
        xcoordinates_start = []
        xcoordinates_end = []
        ycoordinates_adjusted = []
        
        mean_staff_height = int(np.mean(staff_heights))
        vertical_margin = int(mean_staff_height/6) # a staff comprises roughly 6 inter staff line spacings
        horizontal_margin = int(mean_staff_height/2.5)
        
        for counter,ycoord in enumerate(ycoordinates):
            starty = np.maximum(ycoord[0]- vertical_margin,0 )
            endy= np.minimum(ycoord[1]+ vertical_margin,image_height )            
            ycoordinates_adjusted.append([starty,endy])
        for counter,xcoord in enumerate(xcoordinates):     
            staff_start=np.maximum(xcoord[0]-horizontal_margin, left_margin)
            staff_end= np.minimum(xcoord[1]+horizontal_margin, right_margin)
            
            staff_xcoordinates_start,staff_xcoordinates_end=\
            self.slice_staff_horizontally_into_windows(staff_start,staff_end, mean_staff_height,left_margin)               
            
            xcoordinates_start.append(staff_xcoordinates_start)
            xcoordinates_end.append(staff_xcoordinates_end)
        #sort 
        ymins_margins = [y[0] for y in ycoordinates_adjusted]
        sorted_indices= np.argsort(ymins_margins)
        sorted_ycoordinates = np.asarray(ycoordinates_adjusted)[sorted_indices]  
        sorted_xcoordinates_start = np.asarray(xcoordinates_start,dtype=object)[sorted_indices]  
        sorted_xcoordinates_end = np.asarray(xcoordinates_end,dtype=object)[sorted_indices]  
        return sorted_ycoordinates, sorted_xcoordinates_start,sorted_xcoordinates_end,\
            left_margin,right_margin, ornate_elements_boxes, ornate_elements_scores

    def convert_bboxes(self,box_list,x_min,y_min,x_scale_factor,y_scale_factor):        
        #box format: ymin, xmin, ymax, xmax 
        gauge = [y_min,x_min,y_min,x_min]
        corrected_boxes= []
        for box in box_list:            
            rescaled_box = box*[y_scale_factor,x_scale_factor,y_scale_factor,x_scale_factor]
            corrected_box= rescaled_box + gauge
            corrected_boxes.append(corrected_box)        
        return corrected_boxes

    def convert_dict(self,dict, x_min,y_min,x_scale_factor,y_scale_factor):
        dict['boxes']= self.convert_bboxes( dict['boxes'], x_min,y_min,x_scale_factor,y_scale_factor)    
        return dict
         
    def get_staff_box_dict(self,image_path,output_folder,image_height,image_width) :   
        img_name  = image_path.split('/')[-1]
        vertical_flip = False
        if 'verticalflip' in img_name:
            vertical_flip = True
        detection = os.path.join(output_folder,img_name.split('.jp')[0]+'.txt')
        xmin = 0
        ymin =0
        x_scale_factor=  image_width
        y_scale_factor =  image_height    
        inference_dictionary= self.get_dict(detection,vertical_flip)
        output_dict= self.convert_dict(inference_dictionary,xmin,ymin,x_scale_factor,y_scale_factor)            
        return output_dict         
    
    
    def get_output_path_staffs(self,img_name,save_dir,crop_ymin, crop_height):
        save_dir= os.path.join(save_dir, img_name)      
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir) 
        file_name = str(crop_ymin)+'_'+str(crop_height)+'.pickle'        
        save_path = os.path.join(save_dir,file_name)
        return save_path

    def get_output_path_ornate_elements_and_margins(self,image_path,save_dir):
        ext= image_path.split('/')[-1].split('.j')[0]
        save_dir= os.path.join(save_dir, ext)      
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir) 
        file_name = 'ornate_elements_and_margins.pickle'        
        save_path = os.path.join(save_dir,file_name)
        return save_path    

    def get_lig_clusters(self,label_ids,scores):
        low_prob_longa_breve= ['longa', 'breve','colored longa','colored breve']
        low_prob_ids = [label_dict[key] for key in low_prob_longa_breve]
        prob_threshold = 0.5
        
        clusters = []
        new_cluster = True
        i = 0
        while i < len(label_ids):
            cluster =[ ]
            for j in range(i, len(label_ids)):
                label_id = label_ids[j]
                if label_id in ligature_ids:
                    cluster.append(j)
                if scores[j] < prob_threshold and label_id in low_prob_ids:
                    cluster.append(j)
                else: 
                    if len(cluster) != 0:
                        clusters.append(cluster)
                    break
            i = j+1
        if len(cluster) != 0:
            clusters.append(cluster)
        return clusters          

    def write_lig_slices(self,staff_dict,img_path,crop_ymin, crop_height,staffn):
       
        if staff_dict != None:
            xmins = [box[1] for  box in  staff_dict['boxes']]
            sorted_indices_xmins = np.argsort(xmins)
            xmins= np.asarray(xmins)[sorted_indices_xmins]
            xmaxs = [box[-1] for  box in  staff_dict['boxes']]
            xmaxs = np.asarray(xmaxs)[sorted_indices_xmins]
            label_ids = staff_dict['label_ids']
            scores= staff_dict['scores']
            
            margin = int(crop_height/3)
                         
            # get ligature clusters 
            clusters= self.get_lig_clusters(label_ids,scores)
            imgc= cv2.imread(img_path)
            img_name= img_path.split('/')[-1].split('.jp')[0]
            
            lig_image_crops = []
            # get cluster crops:
            for cluster in clusters:
                if len(cluster) != 0:                    
                    lig_cluster_xmin =  xmins[cluster[0]]
                    lig_cluster_xmax =  xmaxs[cluster[-1]]
                    lig_cluster_xmin = int(np.maximum(0, lig_cluster_xmin- margin))
                    lig_cluster_xmax = int(np.minimum(imgc.shape[1],lig_cluster_xmax+ margin))
                    staff_crop = imgc[crop_ymin:crop_ymin + crop_height,lig_cluster_xmin :lig_cluster_xmax]

                    
                    crop_width = str(lig_cluster_xmax-lig_cluster_xmin)
                    
                    img_name = img_path.split('/')[-1].split('.jp')[0]+'_'+str(staffn)+'_'+str(crop_ymin)+\
                    '_'+str(lig_cluster_xmin)+'_'+str(crop_width)+'_'+str(crop_height)+'.jpeg'
                    write_path = os.path.join(self.ligature_image_slices_dir,img_name)
                    cv2.imwrite(write_path,staff_crop)                      
                    lig_image_crops.append(write_path)

#                     img_name = img_path.split('/')[-1].split('.jp')[0]+'verticalflip'+'_'+str(staffn)+'_'+str(crop_ymin)+\
#                     '_'+str(lig_cluster_xmin)+'_'+str(crop_width)+'_'+str(crop_height)+'.jpeg'
#                     write_path = os.path.join(self.ligature_image_slices_dir,img_name)
#                     cv2.imwrite(write_path,cv2.flip(staff_crop, 0))  
#                     lig_image_crops.append(write_path)                                        
            return lig_image_crops        
        else:
            return []

        
    def slice_images(self,input_image_folder,output_slice_folder):     
        image_paths = glob.glob(self.input_folder+'/'"*.jp*g")
        write_dir = output_slice_folder
        dict_of_text_images = {}
        for img_path in image_paths:    
            dict_of_text_images[img_path]  = []
            SLICE_IMG_FRACTION = 1/10
            img = cv2.imread(img_path)
            img_height,  img_width= img.shape[0], img.shape[1]
            slice_height =int(img_height*SLICE_IMG_FRACTION)
            OVERLAP_FRAC = 1/5

            ycoordinates_start =[0]
            staff_slice_heights = []
            while ycoordinates_start[-1]+ slice_height*(1-OVERLAP_FRAC) <img_height:
                ycoordinates_start.append(int(ycoordinates_start[-1]+ slice_height*(1-OVERLAP_FRAC)))         
                staff_slice_heights.append(slice_height)
            if img_height -slice_height >0 and (img_height -slice_height) not in ycoordinates_start:
                ycoordinates_start[-1] = img_height -slice_height
                staff_slice_heights.append(slice_height)
            write_paths=[]
            Xmin,side_length = 0, img_width
            for ycoord,height in zip(ycoordinates_start,staff_slice_heights):
                img_crop = img[ycoord:ycoord+height,0:img_width]
                img_name = img_path.split('/')[-1].split('.jp')[0]+'_'+str(ycoord)+\
                '_'+str(Xmin)+'_'+str(side_length)+'_'+str(height)+'.jpeg'            
                write_path = os.path.join(write_dir,img_name)  
                dict_of_text_images[img_path].append(write_path)
                cv2.imwrite(write_path,img_crop)       
        return dict_of_text_images           
        
    def infer_text_on_images(self):
        input_image_folder =self.input_folder
        output_slice_folder =  self.text_image_slices_dir
        # write sliced images 
        dict_of_text_images = self.slice_images(input_image_folder,output_slice_folder)
#          /home/yoeri/Work/OMR/experiment_omr/text/sliced_text_images does not exis
        #text inference
        self.inference('text', self.text_image_slices_dir , self.text_slices_inference_dir )          
        
        #collect dictionary
        mode = 'text'
        for image_path in dict_of_text_images :  
            output_dict = None            
            save_dir = os.path.join(self.detection_dicts_dir, image_path.split('/')[-1].split('.jp')[0],mode)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for img_slice in dict_of_text_images[image_path]:
                crop_height= int(img_slice.split('_')[-1].split('.jpeg')[0])
                crop_width = int(img_slice.split('_')[-2])
                crop_xmin = int(img_slice.split('_')[-3])
                crop_ymin = int(img_slice.split('_')[-4])                     
                img_slice_path=\
                os.path.join(self.text_slices_inference_dir,img_slice.split('/')[-1].split('.jp')[0]+'.txt')
                if os.path.exists(img_slice_path):
                    vertical_flip = False
                    inference_dictionary= self.get_dict(img_slice_path,vertical_flip)
                    img_name = img_slice.split('/')[-1].split(str(crop_ymin)+'_'+str(crop_xmin)+\
                                                               '_'+str(crop_width)+'_'+str(crop_height))[0]
                    if output_dict != None: 
                        inference_dictionary= self.convert_dict(inference_dictionary,crop_xmin,crop_ymin,crop_width,crop_height)                              
                        output_dict= self.merge_dict(output_dict,inference_dictionary)
                    else:       
                        output_dict= self.convert_dict(inference_dictionary,crop_xmin,crop_ymin,crop_width,crop_height)
            if output_dict != None:
                save_path = os.path.join(save_dir,str(crop_ymin)+'_'+str(crop_height)+'.pickle') 
                with open(save_path, 'wb') as handle:
                        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)        


    def collect_results_inference_to_pickle(self,dictionary_of_dictionaries_containing_images_per_staff,mode):        
        if mode == 'ligatures':                                
            mode_inference_dir = self.ligature_slices_inference_dir
        if mode == 'beginning':           
            mode_inference_dir = self.begin_slices_inference_dir
        if mode == 'end':           
            mode_inference_dir = self.end_slices_inference_dir     
        if mode == 'rare':
            mode_inference_dir = self.sliced_staff_inference_dir_rare    
            
        for image_path in dictionary_of_dictionaries_containing_images_per_staff:  
            save_dir = os.path.join(self.detection_dicts_dir, image_path.split('/')[-1].split('.jp')[0],mode)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for staff_dict_number in  dictionary_of_dictionaries_containing_images_per_staff[image_path].keys():
                output_dict = None    
                for staff_slice in  dictionary_of_dictionaries_containing_images_per_staff[image_path][staff_dict_number]:
                    crop_height= int(staff_slice.split('_')[-1].split('.jpeg')[0])
                    crop_width = int(staff_slice.split('_')[-2])
                    crop_xmin = int(staff_slice.split('_')[-3])
                    crop_ymin = int(staff_slice.split('_')[-4])                     
                    staff_slice_path=\
                    os.path.join(mode_inference_dir,staff_slice.split('/')[-1].split('.jp')[0]+'.txt')
                    if os.path.exists(staff_slice_path):
                        vertical_flip= False
                        if 'verticalflip' in staff_slice:
                            vertical_flip = True
                        inference_dictionary= self.get_dict(staff_slice_path,vertical_flip)
                        img_name = staff_slice.split('/')[-1].split('_'+staff_dict_number+'_'+str(crop_ymin)+'_'+str(crop_xmin)+\
                                                                   '_'+str(crop_width)+'_'+str(crop_height))[0]
                        if output_dict != None: 
                            inference_dictionary= self.convert_dict(inference_dictionary,crop_xmin,crop_ymin,crop_width,crop_height)                              
                            output_dict= self.merge_dict(output_dict,inference_dictionary)
                        else:       
                            output_dict= self.convert_dict(inference_dictionary,crop_xmin,crop_ymin,crop_width,crop_height)
                if output_dict != None:
                    save_path = os.path.join(save_dir,str(crop_ymin)+'_'+str(crop_height)+'.pickle') 
                    with open(save_path, 'wb') as handle:
                            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
    def vert_overlap_staff(self,boxes):
        #merge 
        merged_boxes= [] 
        already_merged = []         
        for i in range(len(boxes)):
            j=i
            if  i not in already_merged:
                merged_box = boxes[i]
                while j < len(boxes):
                    if self.vert_overlap_box(boxes[i],boxes[j]) >0.75 and j not in already_merged: 
                        ymin = np.minimum(boxes[i][0],boxes[j][0])
                        ymax = np.maximum(boxes[i][2],boxes[j][2])
                        xmin = np.minimum(boxes[i][1],boxes[j][1])
                        xmax = np.maximum(boxes[i][3],boxes[j][3])
                        merged_box = [ymin,xmin, ymax, xmax]
                        already_merged.append(j)
                    j+=1
                merged_boxes.append(merged_box)
        return merged_boxes    
    
    def rescale_images(self):
        desired_staff_height = 20
        image_paths=glob.glob(self.input_folder+'/'"*.jp*g")
        self.inference('staff', self.input_folder,self.staff_inference_dir)  
        for image_path in image_paths:
            img = cv2.imread(image_path)
            im_height, im_width= img.shape[0],img.shape[1]
            staff_dict= self.get_staff_box_dict(image_path,self.staff_inference_dir ,im_height,im_width)        
            staffs = self.get_staffs_from_dict(staff_dict)  

            staffs,_, _ = self.get_filtered_staffs_and_page_margins(staffs)
            staffs = self.vert_overlap_staff(staffs)        
            staff_heights = []
            for staff in staffs:
                staff_heights.append(-int(staff[0]) + int(staff[2]))  
            
            error_pctg = 0.1 
            estimate_inter_staff_line_spacing = int(np.median(staff_heights)/6)
            low_estimate_inter_staff_line_spacing = int(estimate_inter_staff_line_spacing*(1-error_pctg))
            # a staff comprises roughly 6 inter staff line spacings        
            scale_factor =  desired_staff_height/low_estimate_inter_staff_line_spacing
#             if scale_factor < 2:
            desired_height = int(scale_factor*img.shape[0])
            desired_width = int(scale_factor*img.shape[1])
            resized= cv2.resize(img, (desired_width,desired_height), interpolation=cv2.INTER_CUBIC)   

#             else:     
#                 scale_factor = np.minimum(int(scale_factor),4) #only accepts integer rescaling
#                 sr = dnn_superres.DnnSuperResImpl_create()
#                 # Read the desired model
#                 sr.readModel('./dcnns/rescaling_models/EDSR_x'+str(scale_factor)+'.pb')
#                 sr.setModel("edsr", scale_factor)
#                 # Upscale the image
#                 resized = sr.upsample(img) 

            shutil.copy(image_path, self.non_rescaled_images_dir)
            cv2.imwrite(image_path, resized)
        shutil.rmtree(self.staff_inference_dir, ignore_errors=True)
        os.makedirs(self.staff_inference_dir) 
            
    def object_detection(self):        
        #collect images
        image_paths=glob.glob(self.input_folder+'/'"*.jpg")
        image_paths+=glob.glob(self.input_folder+'/'"*.jpeg")  
        length = len(image_paths)
        print('Object detection will be performed on {}'.format(length), 'image(s)' )
        
        #rescale images if need be
        self.rescale_images()
    
        #infer text on images
        self.infer_text_on_images()
        #staff inference
        self.inference('staff', self.input_folder,self.staff_inference_dir)  

        dict_of_image_crop_dictionaries = {}
        dictionary_of_dictionaries_containing_beginnings_per_staff = {}
        dictionary_of_dictionaries_containing_endings_per_staff = {}
        for image_path in tqdm(image_paths):
            #staff inference
            dict_img = {}
            begin_dict_img = {}
            end_dict_img = {}
            dict_of_image_crop_dictionaries[image_path]=dict_img
            dictionary_of_dictionaries_containing_beginnings_per_staff[image_path] = begin_dict_img
            dictionary_of_dictionaries_containing_endings_per_staff[image_path] = end_dict_img
            
            img_name = image_path.split('/')[-1].split('.jp')[0]
            output_folder_staffs = os.path.join(self.staff_inference_dir,img_name)
            img = cv2.imread(image_path)
            im_height, im_width= img.shape[0],img.shape[1]
            staff_dict= self.get_staff_box_dict(image_path,self.staff_inference_dir ,im_height,im_width)

            # infer music_symbols per staff
            ycoordinates, xcoordinates_start,xcoordinates_end, left_margin, right_margin,ornate_elements_boxes,\
            ornate_elements_scores= self.get_stave_crop_coordinates(staff_dict,im_height)
            ornate_elements_and_margins_dict = {}
            ornate_elements_and_margins_dict['ornate_element_boxes']= ornate_elements_boxes
            ornate_elements_and_margins_dict['ornate_element_scores']= ornate_elements_scores
            ornate_elements_and_margins_dict['left_margin'] = left_margin
            ornate_elements_and_margins_dict['right_margin'] = right_margin
            with open(self.get_output_path_ornate_elements_and_margins(image_path, self.detection_dicts_dir), 'wb') as handle:
                    pickle.dump(ornate_elements_and_margins_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)        
            #object detection on staffs
            for i in range(len(ycoordinates)):
                dict_img[str(i)]= []
                begin_dict_img[str(i)]= []
                end_dict_img[str(i)]= []
                Ymin = ycoordinates[i][0]
                Ymax= ycoordinates[i][1]
                for j  in range(0,len(xcoordinates_start[i])):
                    Xmin =xcoordinates_start[i][j]                    
                    Xmax= xcoordinates_end[i][j]  
                    if  len(img.shape)== 2:
                        image= img[Ymin:Ymax,Xmin:Xmax]                        
                    else:    
                         image= img[Ymin:Ymax,Xmin:Xmax,:]  
                    stave_height = Ymax-Ymin
                    side_length = Xmax -Xmin   
                    img_name = image_path.split('/')[-1].split('.jp')[0]+'_'+str(i)+'_'+str(Ymin)+\
                    '_'+str(Xmin)+'_'+str(side_length)+'_'+str(stave_height)+'.jpeg'
                    slice_path = os.path.join(self.sliced_staff_images_dir ,img_name)
                    inference_on_sclice_path = os.path.join( self.sliced_staff_inference_dir_all,img_name)
                    dict_img[str(i)].append(inference_on_sclice_path)
                    cv2.imwrite(slice_path,image)   
                    if j in [0,1]:
                        slice_path = os.path.join(self.begin_image_slices_dir ,img_name)
                        inference_on_sclice_path = os.path.join( self.sliced_staff_inference_dir_all,img_name)
                        begin_dict_img[str(i)].append(inference_on_sclice_path)
                        cv2.imwrite(slice_path,image)                           
                    if j in [len(xcoordinates_start[i])-1,len(xcoordinates_start[i])-2]:
                        slice_path = os.path.join(self.end_image_slices_dir ,img_name)
                        inference_on_sclice_path = os.path.join( self.sliced_staff_inference_dir_all,img_name)
                        end_dict_img[str(i)].append(inference_on_sclice_path)
                        cv2.imwrite(slice_path,image)                       
        self.inference("all",self.sliced_staff_images_dir,self.sliced_staff_inference_dir_all)                     
        self.inference("rare",self.sliced_staff_images_dir,self.sliced_staff_inference_dir_rare) 
        self.inference("beginning", self.begin_image_slices_dir,self.begin_slices_inference_dir)   
        self.inference("end",self.end_image_slices_dir,self.end_slices_inference_dir)   
        
           # gather all detection per staff of this image              
        dictionary_of_dictionaries_containing_ligatures_per_staff = {}
    
        for image_path in dict_of_image_crop_dictionaries.keys():                     
            dict_of_ligature_images_per_staff= {}
            dictionary_of_dictionaries_containing_ligatures_per_staff[image_path]=\
                                                        dict_of_ligature_images_per_staff
            staff_keys = dict_of_image_crop_dictionaries[image_path].keys()
            for staff in dict_of_image_crop_dictionaries[image_path]:
                output_dict = None            
                for staff_slice in dict_of_image_crop_dictionaries[image_path][staff]:
                    crop_height= int(staff_slice.split('_')[-1].split('.jpeg')[0])
                    crop_width = int(staff_slice.split('_')[-2])
                    crop_xmin = int(staff_slice.split('_')[-3])
                    crop_ymin = int(staff_slice.split('_')[-4]) 
                    detected_symbols = staff_slice.split('.jpeg')[0]+'.txt'
                    if os.path.exists(detected_symbols):
                        vertical_flip = False
                        inference_dictionary= self.get_dict(detected_symbols,vertical_flip)
                        img_name = staff_slice.split('/')[-1].split('_'+staff+'_'+str(crop_ymin)+'_'+str(crop_xmin)+\
                                                                   '_'+str(crop_width)+'_'+str(crop_height))[0]
                        if output_dict != None: 
                            inference_dictionary= self.convert_dict(inference_dictionary,crop_xmin,crop_ymin,crop_width,crop_height)                              
                            output_dict= self.merge_dict(output_dict,inference_dictionary)
                        else:                         
                            output_dict= self.convert_dict(inference_dictionary,crop_xmin,crop_ymin,crop_width,crop_height)  
                lig_image_crops =self.write_lig_slices(output_dict,image_path,crop_ymin, crop_height,staff)
                dict_of_ligature_images_per_staff[staff]= lig_image_crops                
              
                with open(self.get_output_path_staffs(img_name, self.detection_dicts_dir,crop_ymin, crop_height), 'wb') as handle:
                        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
            dictionary_of_dictionaries_containing_ligatures_per_staff[image_path] = dict_of_ligature_images_per_staff     
            
                # Detect ligatures, write_images_to_file and perform ligature inference, 
        self.inference('ligature',self.ligature_image_slices_dir, self.ligature_slices_inference_dir)   
        
        self.collect_results_inference_to_pickle(dict_of_image_crop_dictionaries,'rare')        
        self.collect_results_inference_to_pickle(dictionary_of_dictionaries_containing_ligatures_per_staff,'ligatures')
        self.collect_results_inference_to_pickle(dictionary_of_dictionaries_containing_beginnings_per_staff,'beginning')
        self.collect_results_inference_to_pickle(dictionary_of_dictionaries_containing_endings_per_staff,'end') 
