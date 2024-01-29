import glob 
import numpy as np
import cv2 
import os
import sys
import random
sys.path.append('./data_creation/')
sys.path.append('./post_dcnn_inference/')
from get_positions_on_staff import *
from siamese_nn import *
from rest_classifier_nn import *
from get_label_dict import get_dict_of_label_ids
label_dict= get_dict_of_label_ids()
inv_label_dict = {v: k for k, v in label_dict.items()}
super_class_dichotomies= {'imaj': 'imin', 'imin': 'imaj', 'fusa': 'semifusa',\
                          'semifusa': 'fusa', 'pmin': 'pmaj', 'pmaj':'pmin'}

ligature_labels = [ 'l1longa', 'l2longa', 'l1colored longa', 'l2colored longa','l1breve',
                    'l2breve', 'l1colored breve','l2colored breve', 'l1semibreve', 'l2semibreve',
                   'l1colored semibreve', 'l2colored semibreve', 'o1longa','o1colored longa', 'o2longa',             
                    'o2colored longa', 'o1breve', 'o1colored breve', 'o2breve', 'o2colored breve', 
                   'o1semibreve', 'o1colored semibreve', 'o2semibreve', 'o2colored semibreve',
                   'l1','l2','o1','o2','colored l1', 'colored o1', 'colored l2', 'colored o2']                  
ligature_ids = [label_dict[label] for label in ligature_labels]

class super_class_parser():
    def __init__(self,reference_input_folder):
        siamese_super_classes = ['fusa','imin', 'pmin']
        self.siamese_super_class_ids = [label_dict[label] for label in siamese_super_classes] 
        self.rest_classifier = rest_nn()
        self.ligature_parser = Ligature_Parsing()
        self.siamese_correcter = Siamese_Corrections(reference_input_folder)
               
    def get_dictionaries_with_superclasses_parsed(self,list_of_staff_dictionaries):
        img_path = list_of_staff_dictionaries[0]['image path']
        staff_parser = Positions_on_staff(img_path)  
        for staff_dict in list_of_staff_dictionaries: 
            new_label_ids = []
            inter_staff_line_spacing=staff_dict['inter_staff_line_spacing']
            for index,(label_id,symbol_box, staff_lines) in \
            enumerate(zip(staff_dict['label_ids'], staff_dict['boxes'],staff_dict['staff_lines_around_music_symbol'])) :
                    if label_id == label_dict['rest']:
                        draw_staff_lines = False
                        img = staff_parser.get_cleaned_staff_exerpt_around_img(img_path,
                                        symbol_box,staff_lines,inter_staff_line_spacing,draw_staff_lines)                       
                        label = self.rest_classifier.predict_label(img)
                        label_id = label_dict[label]
                        staff_dict['label_ids'][index] =label_id
                    if label_id in self.siamese_super_class_ids:
                        draw_staff_lines = True
                        img = staff_parser.get_cleaned_staff_exerpt_around_img(img_path,symbol_box,\
                                                        staff_lines,inter_staff_line_spacing,draw_staff_lines)                                              
                        label = self.siamese_correcter.find_most_likely_super_classes(img,inv_label_dict[label_id])
                        label_id = label_dict[label]
                        staff_dict['label_ids'][index] =label_id                        
            staff_dict = self.ligature_parser.parse_ligatures(staff_dict)
        return  list_of_staff_dictionaries

class Siamese_Corrections(object): 
    """""
    Untangle superclasses: fusa -> {fusa, semifusa}; pmin -> {pmin,pmaj}, imin -> {imin,imaj} 
    Still to do create, dictionary of ref images instead of folder with the actual images 
    Perhaps randomly sample per label x % of all elements or a fixed number, things are reaaaaaally slow now
    you could sample x % of each page to get a more representative sample or a fixed amount from each book
    so as to make the vote more democratic and representative?
    """
    def __init__(self,data_source_path):
        self.data_source_path = data_source_path
        self.reference_dict = self.get_reference_dict()
        self.siamese_inferencer= Siamese_inference()
        
    def get_reference_dict(self):
        ref_dict = {}
        for mode in ['train', 'val']:
            data_dir = os.path.join(self.data_source_path,mode)
            for label in os.listdir(data_dir):
                label_dir = os.path.join(data_dir,label)
                if label not in ref_dict:
                    ref_dict[label] = []
                sub_dirs = os.listdir(label_dir)
                if len(sub_dirs) != 0 and not '.jpeg' in sub_dirs[0]:
                    positions =  sub_dirs
                    for pos in positions:
                        jpegs_data_dir= os.path.join(label_dir, pos)  
                        data = glob.glob(jpegs_data_dir+'/**.jp*g')
                        ref_dict[label]+=data
                else:
                    data = glob.glob(label_dir+'/**.jp*g')
                    ref_dict[label]+=data
            random.shuffle(ref_dict[label])            
        return ref_dict
  
    def find_most_likely_super_classes(self,img,super_class):
        super_class_dichotomies= {'imaj': 'imin', 'imin': 'imaj', 'fusa': 'semifusa',\
                                  'semifusa': 'fusa', 'pmin': 'pmaj', 'pmaj':'pmin'}
        twin = super_class_dichotomies[super_class]
        scores,flipped_scores = {},{}
        if twin in self.reference_dict.keys():
            for label in [super_class, twin]:
                scores[label]=[]
                flipped_scores[label] = []
                for im_path in self.reference_dict[label][0:100]:
                    compare_img = cv2.imread(im_path)
                    score= self.siamese_inferencer.get_similarity_score(img,compare_img)
                    scores[label].append(np.round(score,3))
                    compare_img = cv2.flip(compare_img,1)
                    score= self.siamese_inferencer.get_similarity_score(img,compare_img)
                    scores[label].append(np.round(score,3))            
            median_scores= [np.median(scores[label]) for label in scores]
            untangled_superclass = np.asarray([super_class,twin])[np.argmax(median_scores)]
        else:
            untangled_superclass = super_class
        return untangled_superclass  

    
    
    
class Ligature_Parsing(object):    
    def parse_ligatures_folium(self,list_of_staff_dictionaries):
        for i,staff_dictionary in enumerate(list_of_staff_dictionaries):
                self.parse_ligatures(staff_dictionary)
        return list_of_staff_dictionaries

    def get_lig_clusters(self,staff_dictionary):
        ligature_indices= []
        for index,label_id in enumerate(staff_dictionary['label_ids']):
            if label_id in ligature_ids:
                ligature_indices.append(index)
        if len(ligature_indices) >0:
            all_clusters= [[ligature_indices[0]]]
            for i in range(1,len(ligature_indices)):
                lig_id =staff_dictionary['label_ids'][ligature_indices[i]]
                if 'l1' in inv_label_dict[lig_id] or 'o1' in inv_label_dict[lig_id]:
                    all_clusters.append([ligature_indices[i]])                    
                else:
                    all_clusters[-1].append(ligature_indices[i])
        else:
            all_clusters= []
        all_cleaned_clusters = []
        discarded_indices = []
        for cluster in all_clusters:
            cluster,discarded_index = self.clean_lig_cluster(staff_dictionary,cluster)
            if not discarded_index is None:
                discarded_indices.append(discarded_index)
            all_cleaned_clusters.append(cluster)        
        return  all_cleaned_clusters,discarded_indices
    
    def clean_lig_cluster(self,staff_dict,cluster):
        if len(cluster)>2 and\
        self.compatible_ligature_ids(staff_dict['label_ids'][cluster[0]],staff_dict['label_ids'][cluster[1]]) and\
            self.hor_overlap(staff_dict['boxes'][cluster[1]],staff_dict['boxes'][cluster[0]]) > 0.5:
                return  [cluster[0]]+ cluster[2::], cluster[1]       
        else:
            return cluster, None
        
    def compatible_ligature_ids(self,lig_id1,lig_id2):
        label_lig_id1 = inv_label_dict[lig_id1]
        label_lig_id2 = inv_label_dict[lig_id2]
        return ('l1' in label_lig_id1 and 'l2' in label_lig_id2)         
    
    def hor_overlap(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = np.maximum(boxA[1], boxB[1])
        xB = np.minimum(boxA[3], boxB[3])

        boxA_width = boxA[3]-boxA[1]
        # compute the area of intersection rectangle
        intersection = np.abs(np.maximum(xB - xA, 0))/boxA_width
        return intersection                


    def create_note(self,label, color):
        if color :
            return 'colored '+label
        else:
            return label
   
    def Apel_Ligature_heuristic(self, lig_notes, left_stems, right_stems, stave_positions ):
        #lig_note_format [id, color,stem_up_down, stem_left_right]
        #assumes ordered list, left to right
        constituent_notes = []
        for i in range(len(lig_notes)):
            note = inv_label_dict[lig_notes[i]]
            stave_pos = stave_positions[i]
            color = 'color' in note      
            right_stem = right_stems[i]
            left_stem = left_stems[i]
            if right_stem != 'n' and 'o1' not in note and 'o2' not in note:
                constituent_notes.append(self.create_note('longa', color))
            elif i ==0:
                if left_stem == 'u':
                        constituent_notes.append(self.create_note('semibreve', color))
                elif left_stem == 'd':
                        constituent_notes.append(self.create_note('breve', color))
                else:
                    subsequent_note_stave_pos = stave_positions[1]

                    if stave_pos<= subsequent_note_stave_pos:
                        constituent_notes.append(self.create_note('longa', color ))
                    else:
                        constituent_notes.append(self.create_note('breve', color))

            elif i >=1 and i <  len(lig_notes)-1:        
                if i ==1 and  left_stems[0]== 'u':
                    constituent_notes.append(self.create_note('semibreve', color ))
                else: 
                    constituent_notes.append(self.create_note('breve', color))

            elif i == len(lig_notes)-1:
                if 'o2' not in note: 
                    if not (i == 1 and left_stems[0]== 'u'):
                        previous_note_stave_pos = stave_positions[i-1]                
                        if stave_pos <= previous_note_stave_pos and right_stem != 'd':                 
                            constituent_notes.append(self.create_note('breve', color))
                        else:    
                            constituent_notes.append(self.create_note('longa', color ))
                    else:
                        constituent_notes.append(self.create_note('semibreve', color))
                else:                       
                    if i == 1 and left_stems[i-1]== 'u':
                        constituent_notes.append(self.create_note('semibreve', color))  
                    else:  
                        constituent_notes.append(self.create_note('breve', color )) 
        return constituent_notes
    
    def binarize_bbx_img(self,img,inter_staff_line_spacing, thresh):
        if inter_staff_line_spacing %2 ==0:
            inter_staff_line_spacing -=1          
        return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                     cv2.THRESH_BINARY,inter_staff_line_spacing,thresh)  
    
    def get_vertical_lines(self,image, inter_staff_line_spacing,bin_thresh):
        temp_lines = []
        bin_thresh = 6
         # denoise without affecting stems       
        sigma_value = 0
        minimal_stem_length= int(1.3*inter_staff_line_spacing)
        kernel_height = np.minimum(minimal_stem_length,image.shape[0])
        minimal_stem_width = 1 
        if kernel_height %2 ==0:
            kernel_height -=1              
        kernel_size = (kernel_height ,minimal_stem_width)
        blur = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_value, sigmaY=sigma_value)

        image = cv2.divide(image, blur, scale=255)
        image = self.binarize_bbx_img(image,inter_staff_line_spacing,bin_thresh)  

        estimated_stem_thickness = np.maximum(int(inter_staff_line_spacing/20),1)
         # apply morphology operator to find stems       
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (estimated_stem_thickness,int(inter_staff_line_spacing*0.2)))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        # apply Canny edge detection
        edges = cv2.Canny(image,70,70,apertureSize = 7 )
        # apply Hough line algorithm
        for rho_parameter in range(1,6):
            for threshold in range(0,20):
                lines= cv2.HoughLinesP(image=edges,rho = rho_parameter,theta = 1*np.pi/720, threshold= threshold,\
                            minLineLength=inter_staff_line_spacing,maxLineGap=(inter_staff_line_spacing/4))
                if not isinstance(lines, type(None)):
                    for line in lines:
                        temp_lines.append(list(line[0]))
            lines = temp_lines

        all_lines = []
        if not (lines is None):
            sorted_indices = np.argsort([line[0] for line in lines])
            lines = np.asarray(lines)[sorted_indices]
            all_lines = []
            for line in lines:
                if line[1]> line[3]:
                        sorted_line = [line[0],line[3],line[2],line[1]]
                else:
                        sorted_line =[line[0],line[1],line[2],line[3]]
                vertical_line_length = sorted_line[3] - sorted_line[1] 
                if sorted_line[0] == sorted_line[2]:
                    gradient = 0
                else:
                    gradient =np.abs((sorted_line[3]-sorted_line[1])/(sorted_line[2]-sorted_line[0])) 
                if sorted_line not in all_lines and vertical_line_length > inter_staff_line_spacing and\
                        np.rad2deg(np.arctan(gradient)) >= 60:
                    all_lines.append(sorted_line)
            if len(all_lines) !=0:
                collected_lines = [all_lines[0]]
                for line in all_lines:
                    added = False
                    for i,cline in enumerate(collected_lines):
                        if self.one_D_overlap(line[0],line[2], cline[0],cline[2]) and self.one_D_overlap(line[1],line[3],cline[1],cline[3]): 
                            combined_line = self.combine(line,cline)
                            collected_lines[i] = combined_line
                            added = True
                    if not added:
                        collected_lines.append(line)
                all_lines = collected_lines
        return all_lines  

    def parse_ligatures(self,staff_dictionary):
                    
        lig_clusters, discarded_indices=self.get_lig_clusters(staff_dictionary)       

        inter_staff_line_spacing = staff_dictionary['inter_staff_line_spacing']
        img_path= staff_dictionary['image path']
        grayscale = cv2.imread(img_path,0) 
            
        for cluster in lig_clusters:
            left_stems = []
            right_stems = []
            stave_line_positions= []
            lig_notes  = []
            for i, index in enumerate(cluster): 
                box= staff_dictionary['boxes'][index]
                label_id = staff_dictionary['label_ids'][index]
                pos = staff_dictionary['staff_positions'][index]
                
                staff_lines = staff_dictionary['staff_lines_around_music_symbol'][index]
                inter_staff_line_spacing = staff_dictionary['inter_staff_line_spacing'] 
                if i == 0 and len(cluster) >1:
                    first_el_box_abs_pos = staff_dictionary['staff_lines_around_music_symbol'][index][pos]
                    second_el_pos = staff_dictionary['staff_positions'][cluster[1]]
                    second_el_box_abs_pos = staff_dictionary['staff_lines_around_music_symbol'][cluster[1]][second_el_pos]
                    if  box[0] < first_el_box_abs_pos -2*inter_staff_line_spacing and box[0] < second_el_box_abs_pos - 2*inter_staff_line_spacing:
                        left_stem = 'u'
                        right_stem = 'n'
                    else:
                        
                        abs_positions = [staff_dictionary['staff_lines_around_music_symbol'][index][pos],\
                                         second_el_box_abs_pos]
                        left_stem,right_stem = self.infer_stems_from_vertical_lines_in_box(inter_staff_line_spacing,
                                        grayscale,staff_lines,pos,abs_positions, box,0)
                else:
                    abs_positions = [staff_dictionary['staff_lines_around_music_symbol'][index][pos],0]
                    if len(cluster)>1 and i <len(cluster)-1:
                        next_pos = staff_dictionary['staff_positions'][cluster[i+1]]
                        second_el_box_abs_pos = staff_dictionary['staff_lines_around_music_symbol'][cluster[i+1]][next_pos]
                        abs_positions[1]= second_el_box_abs_pos
                    
                    left_stem,right_stem = self.infer_stems_from_vertical_lines_in_box(inter_staff_line_spacing,
                                            grayscale,staff_lines,pos,abs_positions, box,i)
                left_stems.append(left_stem)
                right_stems.append(right_stem)
                lig_notes.append(label_id)
                stave_line_positions.append(pos)

            if len(lig_notes) >1:
                parsed_notes=self.Apel_Ligature_heuristic(lig_notes,left_stems, right_stems, stave_line_positions)
                for el,parsed_note in zip(cluster,parsed_notes):
                    if inv_label_dict[staff_dictionary['label_ids'][el]][0:2] in ['l1','o1','l2','o2' ]:
                        parsed_note = inv_label_dict[staff_dictionary['label_ids'][el]][0:2]+parsed_note
                    else:
                        parsed_note= inv_label_dict[staff_dictionary['label_ids'][el]][-2:]+parsed_note
                    if parsed_note in label_dict.keys():
                        staff_dictionary['label_ids'][el] = label_dict[parsed_note] 
        if len(discarded_indices)>0:
            kept_indices = [index for index in range(len(staff_dictionary['label_ids'])) if index not in discarded_indices]
            for key in ['boxes', 'label_ids', 'scores','staff_lines_around_music_symbol', 'staff_positions']:  
                staff_dictionary[key] = np.asarray(staff_dictionary[key])[kept_indices]              
              
        return staff_dictionary     
    
    def infer_stems_from_vertical_lines_in_box(self,inter_staff_line_spacing,grayscale, staff_lines,pos,abs_positions,
                                               box, cluster_number):
        lig_image_ymin = int(np.maximum(0,staff_lines[pos]-2*inter_staff_line_spacing))                     
        lig_image_ymax = int(np.minimum(grayscale.shape[0],staff_lines[pos]+2*inter_staff_line_spacing))
        if  cluster_number == 0:
            lig_image= grayscale[lig_image_ymin:lig_image_ymax, box[1]:box[3]]
            ref_ymin = lig_image_ymin
        else: 
            lig_image= grayscale[box[0]:box[2], box[1]:box[3]]
            
            ref_ymin = box[0]
        lig_image_width = box[3]-box[1]
        symbol_pos_in_box =staff_lines[pos]- lig_image_ymin 
        ad_thresh = 7
        vertical_lines = self.get_vertical_lines(lig_image,inter_staff_line_spacing,ad_thresh)
        abs_positions = [abs_positions[0]-lig_image_ymin, abs_positions[1]-lig_image_ymin]
        
        left_stem, right_stem= self.get_stem_orientation(vertical_lines,abs_positions,\
                                                         inter_staff_line_spacing,lig_image_width)
        return left_stem,right_stem        
        
    def get_stem_orientation(self,lines,positions,inter_staff_line_spacing,im_width):
        left_stems ,right_stems_down = [],[]
        minimal_stem_height_from_position = 1.33*inter_staff_line_spacing
        pos = positions[0]
        next_pos = positions[1]

        for line in lines: 

            if line[0] <= im_width/2 or line[2] <= im_width/2:
                height_above_stave_line_pos = pos-line[1]  
                height_below_stave_line_pos = line[3]-pos               
                if height_below_stave_line_pos > minimal_stem_height_from_position :
                    left_stems.append([height_below_stave_line_pos,'d'])
                if height_above_stave_line_pos > minimal_stem_height_from_position :
                    left_stems.append([height_above_stave_line_pos,'u'])                                                 
            else:
                height_below_stave_line_pos = line[3]-pos
                if height_below_stave_line_pos >  minimal_stem_height_from_position and\
                    height_below_stave_line_pos> positions[1]-positions[0]:
                    right_stems_down.append(height_below_stave_line_pos)
        left_stem, right_stem = 'n','n'
        left_stem_length, right_stem_length = 0,0
        left_stem_or = 'n'
        if len(right_stems_down) !=0:
            lengths=  right_stems_down
            sorted_indices=np.argsort(lengths)
            right_stem_length = np.asarray(right_stems_down)[sorted_indices[-1]]
        if len(left_stems)!=0:
            lengths= [left_stem[0] for left_stem in left_stems]
            sorted_indices=np.argsort(lengths)
            left_stem = np.asarray(left_stems)[sorted_indices[-1]]
            left_stem_length = int(float(left_stem[0]))
            left_stem_or = left_stem[1]
            left_stem = left_stem_or
        if right_stem_length>0 and right_stem_length  > left_stem_length:
            left_stem, right_stem = 'n', 'd'
        elif len(left_stems)!=0 and left_stem_length > right_stem_length:
            left_stem =left_stem_or
            right_stem = 'n'
        return left_stem,right_stem   

    def one_D_overlap(self,A1,A2,B1,B2):
        return np.minimum(A2,B2)- np.maximum(A1,B1) >=0

    def combine(self,line1,line2):
        A1 = np.minimum(line1[0],line2[0])
        A2 = np.minimum(line1[1],line2[1])
        A3 = np.maximum(line1[2],line2[2])
        A4 = np.maximum(line1[3],line2[3])    
        return [A1,A2,A3,A4] 
