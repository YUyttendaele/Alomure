import os 
import numpy as np
import sys
sys.path.append('../data_creation')
from get_label_dict import get_dict_of_label_ids 
from stave_line_detection import  get_stave_line_candidates
label_dict= get_dict_of_label_ids()

class Inter_Staff_Line_Spacing_Finder():
    """
    Estimates the inster staff line spacing for each staff; makes use of the bounding box height
    of common music symbols.
    
    """
    def __init__(self, list_of_dictionaries,grayscale):  
        self.list_of_dictionaries = list_of_dictionaries
        self.grayscale =grayscale
        self.identification_bound = 0.35 # as a fraction of the inter staff line spacing
        self.adaptive_thresholding_constant = 6

    def add_inter_staff_spacing_estimated_to_dicts(self):
        inter_staff_line_spacings= []
        for staff_dict in self.list_of_dictionaries:
            upper_staff_margin = staff_dict['ymin'] 
            lower_staff_margin = staff_dict['staff_height'] +staff_dict['ymin'] 
            inter_staff_line_spacing =self.find_inter_staff_line_spacing(staff_dict['boxes'],\
                                    staff_dict['label_ids'],upper_staff_margin, lower_staff_margin)                     
            inter_staff_line_spacings.append(inter_staff_line_spacing)           
        for i in range(len(inter_staff_line_spacings)):
            self.list_of_dictionaries[i]['inter_staff_line_spacing'] = inter_staff_line_spacings[i]
        return self.list_of_dictionaries  
       
    def find_inter_staff_line_spacing(self,filtered_boxes ,filtered_classes,upper_staff_margin, lower_staff_margin):   
        four_staff_line_symbols = ['minim','semiminim','fusa','rest', 'semibreve rest', 'breve rest',
                                  'longa rest', 'fusa rest', 'minim rest', 'semifusa', 'barline',
                                   'colored longa','longa']
        two_staff_line_symbols = ['breve','colored breve', 'semibreve','colored semibreve',]
        four_staff_line_symbols_ids = [label_dict[x] for x in  four_staff_line_symbols]
        two_staff_line_symbols_ids = [label_dict[x] for x in two_staff_line_symbols]
        sample_stave_differences= []
        for box, class_type in zip(filtered_boxes, filtered_classes):  
            if class_type in four_staff_line_symbols_ids:
                box_height= box[2]-box[0]
                staff_fraction = 1
                inter_staff_spacing_estimate = np.round(box_height/(4*staff_fraction)) 
            elif class_type in two_staff_line_symbols_ids:      
                box_height= box[2]-box[0]
                staff_fraction = 2/4
            else:
                # estimate staff height between upper and lower staff margin using boxes as a prior
                box_height = lower_staff_margin-upper_staff_margin
                staff_fraction = 2
            inter_staff_spacing_estimate = int(np.round(box_height/(4*staff_fraction))) 
            inference_box = [upper_staff_margin, box[1]-int(0.5*inter_staff_spacing_estimate),
                             lower_staff_margin, box[3]+int(0.5*inter_staff_spacing_estimate)]
            staves= get_stave_line_candidates(self.grayscale,inference_box,inter_staff_spacing_estimate)                                        
            if len(staves)>=2:
                line_spacings =self.spacings(staves)
                for line_spacing in line_spacings:
                    if np.abs(1- (line_spacing /inter_staff_spacing_estimate)) <=self.identification_bound:
                        sample_stave_differences.append(line_spacing)
        sample_stave_differences =np.sort(sample_stave_differences)
        # get better estimates using inter_staff_spacing_estimate as a prior 
         
        if len(sample_stave_differences) >0:
            inter_staff_spacing_estimate = int(np.median(sample_stave_differences))
            sample_stave_differences= []

            list_of_stave_lines_per_stave_slice= []
            adaptive_thresholds = []
            for box, class_type in zip(filtered_boxes, filtered_classes):  
                inference_box = [upper_staff_margin, np.maximum(box[1]-int(0.5*inter_staff_spacing_estimate),0),
                lower_staff_margin, np.minimum(box[3]+int(0.5*inter_staff_spacing_estimate),self.grayscale.shape[1])]                
                staves= get_stave_line_candidates(self.grayscale,inference_box,inter_staff_spacing_estimate)                           
                if len(staves)>=2:
                    line_spacings =self.spacings(staves)
                    for line_spacing in line_spacings:
                        if np.abs(1- (line_spacing /inter_staff_spacing_estimate)) <=self.identification_bound:
                            sample_stave_differences.append(line_spacing)
            sample_stave_differences =np.sort(sample_stave_differences) 
            std = np.std(sample_stave_differences)
            median_inter_staff_line_spacing = np.median(sample_stave_differences)
            filtered_sample_stave_differences= []
            for stave_difference in sample_stave_differences:
                if np.abs(stave_difference- median_inter_staff_line_spacing)<= std:
                    filtered_sample_stave_differences.append(stave_difference)
            return int(np.round(np.median(filtered_sample_stave_differences)))  
        else: 
            return None       
    
    def spacings(self,candidate_stave_lines: list):
        spacings= []
        for k in range(1, len(candidate_stave_lines)):
            height_difference= candidate_stave_lines[k]-candidate_stave_lines[k-1]
            spacings.append(height_difference)
        return spacings 
