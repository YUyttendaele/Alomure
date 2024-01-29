import numpy as np
import sys
sys.path.append('../data_creation/')
from get_label_dict import get_dict_of_label_ids
sys.path.append('../post_dcnn_inference/')
from get_positions_on_staff import *
from symbol_classifier import *
from pitch_nn import *
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
ligature_ids = [label_dict[label] for label in ligature_labels]

class clean_folium_heuristics():
    def __init__(self, folium):
        self.folium = folium 
        self.img_path= self.folium[0]['image path']
        self.position_on_staff_finder = Positions_on_staff(self.img_path)
        self.position_nn= staff_position_nn()
        self.classifier = classification_nn()

    def correct_folium(self):
        folium = self.correct_common_symbols_between_staffs(self.folium)
        folium  = self.remove_false_positive_dots(folium)
        folium = self.isolated_ligature_fix(folium)
        folium = self.clef_presence_check_and_fix(folium)      
        folium = self.perfect_prolation_semibreve_swapper(folium)#
        folium = self.no_symbols_before_clef_no_symbols_after_barline_and_custos(folium)
#        folium = self.remove_position_outliers(folium)
        folium = self.correct_flats(folium)
        folium = self.correct_custos_position(folium)
        return folium   
    

    def remove_false_positive_dots(self,folium):  


        dot_scores = []
        for staff_dict in folium:
            discarded_dot_indices= []
            inter_staff_line_spacing = staff_dict['inter_staff_line_spacing']
            for box_index, (label_id,score) in enumerate(zip(staff_dict['label_ids'],staff_dict['scores'])):
                if label_id == label_dict['dot']:        
                    dot_scores.append(score)
        dot_mean = np.quantile(dot_scores,0.5)

        for staff_dict in folium:
            discarded_dot_indices= []
            inter_staff_line_spacing = staff_dict['inter_staff_line_spacing']
            for box_index, (label_id, box,staff_lines,score) in enumerate(zip(staff_dict['label_ids'], staff_dict['boxes'],staff_dict['staff_lines_around_music_symbol'],staff_dict['scores'])):
                if label_id == label_dict['dot']:
                    box_middle = 0.5*(box[2] + box[0])                   
                    image_path = staff_dict['image path']

                    most_proximous_line_dist = 10e5
                    most_proximous_line_index = -2
                    for index,line in enumerate(staff_lines):
                        if np.abs(box_middle-line)< most_proximous_line_dist:
                            most_proximous_line_dist = np.abs(box_middle-line)
                            most_proximous_line_index = index
                    if most_proximous_line_index%2 == 1 and most_proximous_line_dist< 0.25*inter_staff_line_spacing\
                        and  box[2]-box[0]>1.3*inter_staff_line_spacing :
                        discarded_dot_indices.append(box_index)

            if len(discarded_dot_indices)>0:
                kept_indices = [index for index in range(len(staff_dict['label_ids'])) if index not in discarded_dot_indices]
                for key in ['boxes', 'label_ids', 'scores','staff_lines_around_music_symbol', 'staff_positions']:  
                    staff_dict[key] = np.asarray(staff_dict[key])[kept_indices]              
        return folium
    def correct_common_symbols_between_staffs(self,folium):
        allowed_overlap = 0.5
        for k in  range(len(folium)-1):
            staff_A = folium[k]
            staff_B = folium[k+1]
            discarded_A = []
            discarded_B = []
            for i,(box_A,score_A) in enumerate(zip(staff_A['boxes'],staff_A['scores'] )):
                for j,(box_B,score_B) in enumerate(zip(staff_B['boxes'],staff_B['scores'] )):
                    if self.box_overlap(box_A, box_B) >  allowed_overlap or self.box_overlap(box_B, box_A) > allowed_overlap:
                        if score_A > score_B:
                            discarded_B.append(j)
                        else: 
                            discarded_A.append(i)
            
            if len(discarded_A)>0:
                kept_indices = [index for index in range(len(staff_A['label_ids'])) if index not in discarded_A]
                for key in ['boxes', 'label_ids', 'scores','staff_lines_around_music_symbol', 'staff_positions']:  
                    staff_A[key] = np.asarray(staff_A[key])[kept_indices]                 
            if len(discarded_B)>0:
                kept_indices = [index for index in range(len(staff_B['label_ids'])) if index not in discarded_B]
                for key in ['boxes', 'label_ids', 'scores','staff_lines_around_music_symbol', 'staff_positions']:  
                    staff_B[key] = np.asarray(staff_B[key])[kept_indices]
        return folium
                    
    def box_overlap(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = np.maximum(boxA[1], boxB[1])
        xB = np.minimum(boxA[3], boxB[3])
        yA = np.maximum(boxA[0], boxB[0])
        yB = np.minimum(boxA[2], boxB[2])
        boxA_width = boxA[3]-boxA[1]
        boxA_height = boxA[2]-boxA[0]
        # compute the area of intersection rectangle
        intersection = np.abs(np.maximum(xB - xA, 0))*np.abs(np.maximum(yB -yA, 0))/(boxA_width*boxA_height)
        return intersection                   
                
    def get_lig_cluster(self,staff_dictionary):
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
        return all_clusters
    
    def isolated_ligature_fix(self,folium):
        for staff_dict in folium:
            lig_clusters = self.get_lig_cluster(staff_dict)
            image_path = staff_dict['image path']
            folium_image  = cv2.imread(image_path)
            boxes = staff_dict['boxes']             
            for cluster in lig_clusters:
                if len(cluster) ==1:
                    i= cluster[0]
                    img = folium_image[boxes[i][0]:boxes[i][2],boxes[i][1]:boxes[i][3]]                
                    predicted_label= self.classifier.predict_label(img)[0]  
                    if predicted_label in  ['c clef', 'f clef', 'g clef']:
                        staff_dict['label_ids'][i] = label_dict[predicted_label]
                        if predicted_label != 'g clef':                                                                
                            inter_staff_line_spacing= staff_dict['inter_staff_line_spacing']
                            draw_staff_lines = False
                            staff_lines = staff_dict['staff_lines_around_music_symbol'][i]
                            
                            pitch_img=  self.position_on_staff_finder.\
                            get_cleaned_staff_exerpt_around_img(image_path,boxes[i],staff_lines,\
                                                            inter_staff_line_spacing,draw_staff_lines)
                            staff_dict['staff_positions'][i]=  self.position_nn.predict_clef_position(pitch_img)                            
                        else:
                            staff_dict['staff_positions'][i] = 9
                    staff_dict['label_ids'][i] = label_dict[predicted_label]                    
        return folium 
    
    def clef_presence_check_and_fix(self, folium):
        for staff_dict in folium:
            initial_labels = [inv_label_dict[label_id] for label_id in staff_dict['label_ids'][0:5]]
            clef_found =False
            for label in initial_labels:
                if label in ['c clef', 'f clef', 'g clef']:
                    clef_found = True
            if not clef_found:
                i=0
                image_path = staff_dict['image path']
                folium_image  = cv2.imread(image_path)
                boxes = staff_dict['boxes']
                clef_found = False
                while i < 3 and i < len(staff_dict['staff_lines_around_music_symbol']) and not clef_found:
                    staff_lines = staff_dict['staff_lines_around_music_symbol'][i]
                    symbol_box = [int(staff_lines[2]),boxes[i][1],int(staff_lines[12]),boxes[i][3]]
                    img = folium_image[symbol_box[0]:symbol_box[2],symbol_box[1]:symbol_box[3]]                
                    predicted_labels= self.classifier.predict_label(img)
                    for predicted_label in predicted_labels:
                        if predicted_label in  ['c clef', 'f clef', 'g clef']:

                            staff_dict['label_ids'][i] = label_dict[predicted_label]
                            if predicted_label != 'g clef':                                                                
                                inter_staff_line_spacing= staff_dict['inter_staff_line_spacing']
                                draw_staff_lines = False
                                pitch_img=  self.position_on_staff_finder.\
                                get_cleaned_staff_exerpt_around_img(image_path,symbol_box,staff_lines,\
                                                                inter_staff_line_spacing,draw_staff_lines)
                                staff_dict['staff_positions'][i]=  self.position_nn.predict_clef_position(pitch_img)                            
                            else:
                                staff_dict['staff_positions'][i] = 9
                            clef_found = True
                            break
                    i+=1
        return folium
    
    def perfect_prolation_semibreve_swapper(self,folium):
        image_path = folium[0]['image path']
        for staffn, staff_dict in enumerate(folium):
            for i in range(1,len(staff_dict['label_ids'])):
                if staff_dict['label_ids'][i] in [label_dict['pmin'],label_dict['pmaj']]and\
                staff_dict['label_ids'][i-1] not in beginning_ids:
                        staff_dict['label_ids'][i] = label_dict['semibreve']
                        staff_lines = staff_dict['staff_lines_around_music_symbol'][i]
                        inter_staff_line_spacing= staff_dict['inter_staff_line_spacing']
                        symbol_box = staff_dict['boxes'][i]                        
                        draw_staff_lines = False
                        pitch_img=  self.position_on_staff_finder.\
                        get_cleaned_staff_exerpt_around_img(image_path,symbol_box,staff_lines,\
                                                        inter_staff_line_spacing,draw_staff_lines)
                        staff_dict['staff_positions'][i]= self.position_nn.predict_staff_position(pitch_img)                       
        return folium
    
    def no_symbols_before_clef_no_symbols_after_barline_and_custos(self,folium):  
        lower_sequence_length_bound_after_first_barline =0
        for staffn, staff_dict in enumerate(folium):
            last_index_of_the_staff_beginning  = 5
            discarded_begin_indices = []
            clef_found = False
            for index in range(0, np.minimum(len(staff_dict['label_ids']),last_index_of_the_staff_beginning)):
                if inv_label_dict[staff_dict['label_ids'][index]] in ['c clef', 'f clef', 'g clef']:
                    clef_found = True
                    break
                else: 
                    discarded_begin_indices.append(index)
            if not clef_found:
                discarded_begin_indices = []
            #find index of first custos or barline
            discarded_end_indices = []
            end_ids = [label_dict[label] for label in ['barline','custos']]
            potential_end_index= 0
            while potential_end_index< len(staff_dict['label_ids']) and staff_dict['label_ids'][potential_end_index] not in end_ids:
                potential_end_index +=1
            if potential_end_index !=  len([staff_dict['label_ids']])-1:
                scores = []
                for index  in range(potential_end_index, len(staff_dict['label_ids'])):
                    scores.append(staff_dict['scores'][index])
                if len(scores)>lower_sequence_length_bound_after_first_barline:
                    if np.mean(scores) < np.mean(staff_dict['scores'])-np.std(staff_dict['scores']):
                        discarded_end_indices = list(range(potential_end_index+1, len(staff_dict['label_ids'])))
                else:
                    discarded_end_indices = list(range(potential_end_index+1, len(staff_dict['label_ids'])))
            discarded_indices =  discarded_begin_indices + discarded_end_indices
            if len(discarded_indices)>0:
                kept_indices = [index for index in range(len(staff_dict['label_ids'])) if index not in discarded_indices]
                for key in ['boxes', 'label_ids', 'scores','staff_lines_around_music_symbol', 'staff_positions']:  
                    staff_dict[key] = np.asarray(staff_dict[key])[kept_indices]                               
        return folium

    def remove_position_outliers(self,folium):
        for staff_dict in folium:
            start_labels_with_pos = ['c clef','f clef','g clef', 'flat']
            start_label_ids= [label_dict[label] for label in start_labels_with_pos]
            music_symbol_indices = [index for index in range(len(staff_dict['staff_positions'])) if staff_dict['staff_positions'][index] is not None\
                                   and staff_dict['label_ids'][index] not in start_label_ids]
            index_of_first_sane_melodic_line = 0
            minimum_melodic_line_length = 2
            maximum_allowed_interval = 8
            while index_of_first_sane_melodic_line < len(music_symbol_indices)-minimum_melodic_line_length:
                staff_pos_2_ahead = staff_dict['staff_positions'][music_symbol_indices[index_of_first_sane_melodic_line+2]]
                staff_pos_1_ahead = staff_dict['staff_positions'][music_symbol_indices[index_of_first_sane_melodic_line+1]] 
                staff_pos = staff_dict['staff_positions'][music_symbol_indices[index_of_first_sane_melodic_line]]    
                if not(np.abs(staff_pos_2_ahead-staff_pos_1_ahead) < maximum_allowed_interval and\
                    np.abs(staff_pos_1_ahead-staff_pos) < maximum_allowed_interval):
                    index_of_first_sane_melodic_line +=1
                else:
                    break
            #forwards checking
            discarded_forward_indices = []
            last_sane_staff_pos = staff_dict['staff_positions'][music_symbol_indices[index_of_first_sane_melodic_line]] 
            i  =  index_of_first_sane_melodic_line
            while i <len(music_symbol_indices)-1:
                current_staff_pos =  staff_dict['staff_positions'][music_symbol_indices[i+1]]
                if np.abs(last_sane_staff_pos-  current_staff_pos) > maximum_allowed_interval:
                    discarded_forward_indices.append(music_symbol_indices[i+1])
                else:
                    last_sane_staff_pos  =staff_dict['staff_positions'][music_symbol_indices[i+1]]
                i+=1
            #backwards checking    
            discarded_backward_indices = []
            i  =  index_of_first_sane_melodic_line
            last_sane_staff_pos = staff_dict['staff_positions'][music_symbol_indices[index_of_first_sane_melodic_line]] 
            while i >0:
                current_staff_pos =  staff_dict['staff_positions'][music_symbol_indices[i-1]] 
                if np.abs(last_sane_staff_pos-  current_staff_pos) > maximum_allowed_interval:
                    discarded_backward_indices.append(music_symbol_indices[i-1])
                else:
                    last_sane_staff_pos = staff_dict['staff_positions'][music_symbol_indices[i-1]]

                i-=1
            discarded_indices= discarded_backward_indices+discarded_forward_indices
            kept_indices = [index for index in range(len(staff_dict['label_ids']))\
                if index not in discarded_indices]
            for key in ['boxes', 'label_ids', 'scores','staff_lines_around_music_symbol', 'staff_positions']:  
                staff_dict[key] = np.asarray(staff_dict[key])[kept_indices]
        return folium           
    
    def correct_flats(self,parsed_folium):
            begin_staff_labels = ['imaj', 'pmaj', 'imin', 'pmin', 'imincut', 
                                   'pmincut', '3', '2', '1', 'c clef', 'f clef', 'g clef', 'flat']        
            for staff_dict in parsed_folium:
                last_index_of_the_staff_beginning_upper_bound  = 6
                ids = list(staff_dict['label_ids'])
                last_index_of_the_staff_beginning = 0 
                while last_index_of_the_staff_beginning<last_index_of_the_staff_beginning_upper_bound\
                    and last_index_of_the_staff_beginning < len(staff_dict['label_ids']) and \
                inv_label_dict[staff_dict['label_ids'][last_index_of_the_staff_beginning]] in begin_staff_labels:
                        last_index_of_the_staff_beginning +=1                    
                flat_indices= [index for index in range(len(ids)) if ids[index] == label_dict['flat']]
                discarded_indices = []
                for flat_index in flat_indices:
                    clef_label,clef_pos = None, None
                    flat_pos = staff_dict['staff_positions'][flat_index]
                    for index in range(np.maximum(flat_index-3,0), flat_index):
                        if inv_label_dict[staff_dict['label_ids'][index]] in ['c clef', 'f clef', 'g clef']:
                            clef_label = inv_label_dict[staff_dict['label_ids'][index]]
                            clef_pos = staff_dict['staff_positions'][index]
                    if clef_label != None:
                        corrected_flat_pos = self.correct_flat_around_clef(clef_label,clef_pos, flat_pos)
                        staff_dict['staff_positions'][flat_index] = corrected_flat_pos
                    elif last_index_of_the_staff_beginning < flat_index and flat_index < len(staff_dict['label_ids'])-1:
                        next_pos = staff_dict['staff_positions'][flat_index+1]
                        if next_pos == None:
                            discarded_indices.append(flat_index)
                        elif np.abs(next_pos-flat_pos)>2 and (flat_pos in [0,2] or flat_pos in [12,14]):
                            discarded_indices.append(flat_index)
                if len(discarded_indices)>0:
                    kept_indices= [index for index in range(len(staff_dict['label_ids'])) if index not in discarded_indices]                       
                    for key in ['boxes', 'label_ids', 'scores','staff_lines_around_music_symbol', 'staff_positions']:  
                        staff_dict[key] = np.asarray(staff_dict[key])[kept_indices]
            return parsed_folium         

    def correct_flat_around_clef(self,clef_label,clef_pos, flat_pos):
        if clef_label == 'c clef':
            possible_flat_positions = [clef_pos +1, clef_pos-6]
        elif clef_label == 'f clef':
            possible_flat_positions = [clef_pos +4, clef_pos-3]
        else:
            possible_flat_positions = [7]

        correct_pos = flat_pos
        for possible_flat_position in possible_flat_positions:
            if np.abs(possible_flat_position - flat_pos) <3:
                correct_pos  = possible_flat_position
                break
        return correct_pos     
            
    def correct_custos_position(self,parsed_folium):
        symbols_with_pitch = ['maxima', 'longa', 'colored longa', 'breve', 'colored breve',
                              'semibreve', 'colored semibreve', 'minim', 'semiminim', 'fusa',
                              'semifusa', 'l1longa', 'l2longa', 'l1colored longa', 'l2colored longa', 
                              'l1breve', 'l2breve', 'l1colored breve', 'l2colored breve', 'l1semibreve',
                              'l2semibreve', 'l1colored semibreve', 'l2colored semibreve', 'o1longa',
                              'o1colored longa', 'o2longa', 'o2colored longa', 'o1breve', 'o1colored breve',
                              'o2breve', 'o2colored breve', 'o1semibreve', 'o1colored semibreve', 'o2semibreve']

        for i in range(len(parsed_folium)-1):
            staff_dict = parsed_folium[i]
            if staff_dict['label_ids'][-1] == label_dict['custos']:
                for j,label_id in enumerate(parsed_folium[i+1]['label_ids']):
                    if inv_label_dict[label_id] in symbols_with_pitch:
                        staff_dict['staff_positions'][-1]= parsed_folium[i+1]['staff_positions'][j]
                        break
        return parsed_folium  
