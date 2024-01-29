import numpy as np
import sys
sys.path.append('./scripts_post_dcnn_inference/')
from stave_line_detection import get_stave_line_candidates

import cv2
class Middle_Finder(object): 

    def __init__(self,staff_dict):
        #define constants
        jitter_percentage = 0.2
        inter_staff_line_spacing_error_percentage= 0.15
        error_percentage = jitter_percentage + inter_staff_line_spacing_error_percentage
        self.list_of_staff_lines_per_staff_segment = None
        self.staff_dict = staff_dict
        self.inter_staff_line_spacing = self.staff_dict['inter_staff_line_spacing']
        self.staff_line_identifiability_margin =int(np.round(error_percentage*self.inter_staff_line_spacing))
        self.sampling_frequency = int(np.round((1/3)*self.inter_staff_line_spacing))
        self.slice_width = int((0.8)*self.inter_staff_line_spacing)
        self.upper_staff_margin = self.staff_dict['ymin'] 
        self.lower_staff_margin = self.staff_dict['staff_height'] +self.staff_dict['ymin']        
        
    def get_staff_middle_dict(self):
        middle_candidates_dict = {}
        scores_candidates = []
        index = 0
        valid_indices = 0
        list_of_stave_lines_per_stave_slice, stave_slices=\
                                    self.infer_list_of_detected_stave_lines_in_a_staff_segment()    
        self.list_of_staff_lines_per_staff_segment= list_of_stave_lines_per_stave_slice
        middles, scores = [],[]
        dict_of_middles  = {}
        for  index in range(len(list_of_stave_lines_per_stave_slice)):
            detected_lines_in_histogram = list_of_stave_lines_per_stave_slice[index]
            middles, scores = self.infer_middles_from_detected_lines_in_histogram(detected_lines_in_histogram)       
            dict_of_middles = self.update_dict_of_middles(dict_of_middles, middles,scores,index)
        return dict_of_middles        
        
    def get_list_of_staff_lines_per_staff_slice(self):  
        return self.list_of_staff_lines_per_staff_segment
    
    def update_line_in_dict(self,line_trace,compatible_line_extensions):
        middles = []
        scores = []
        if len(line_trace) >0:
            for i in range(0,np.minimum(len(line_trace),2)):
                middles.append(line_trace[len(line_trace)-1-i][0])
                scores.append(line_trace[len(line_trace)-1-i][1])        
        for line_extension in compatible_line_extensions:
            middles.append(line_extension[0])
            scores.append(line_extension[1])
        to_add_middle, to_add_score, total_score = 0,0,0
        for middle,score in zip(middles, scores):
            to_add_middle += middle*score
            to_add_score += score*score
            total_score +=score
        return int(np.round(to_add_middle/total_score)), int(np.round(to_add_score/total_score))
    
    def update_dict_of_middles(self,dict_of_middles, middles,scores,index):
        #form of  dict key = middle0 , values =[[m00,score0,0],[m01,score1,1]
        middles_added = []
        for middle_in_dict in dict_of_middles:   
            # find closest middle already in dictionary
            compatible_line_extensions = []
            for middle, score, in zip(middles,scores):
                if np.abs(dict_of_middles[middle_in_dict][-1][0]-middle)<= self.staff_line_identifiability_margin:
                        compatible_line_extensions.append([middle,score])
                        middles_added.append(middle)
            line_trace =dict_of_middles[middle_in_dict]
            if len(compatible_line_extensions)!= 0:
                to_add_middle, to_add_score = self.update_line_in_dict(line_trace,compatible_line_extensions)
                dict_of_middles[middle_in_dict].append([to_add_middle, to_add_score,index])
                
        for middle, score, in zip(middles,scores):
            if middle not in middles_added:
                if middle not in dict_of_middles.keys():
                    dict_of_middles[middle] = [[middle,score,index]]
                else:
                    epsilon = 0.01
                    dict_of_middles[middle+epsilon] = [[middle,score,index]]
        return dict_of_middles       
   
    def get_middle(self): 
        dict_of_middles = self.get_staff_middle_dict()
        most_voted_key = self.get_most_voted_key(dict_of_middles)
        middle, middle_index_in_stave_slices = self.get_middle_and_middle_index_within_trace(dict_of_middles,
                                                                                             most_voted_key)
        return dict_of_middles,middle, middle_index_in_stave_slices   
    
    def infer_list_of_detected_stave_lines_in_a_staff_segment(self):
        # get the stave lines of each slice of the partition of the staff width and get the the indices
        # of which slice of the partition corresponds to a box of a music symbol     
        boxes = self.staff_dict['boxes']
        grayscale =  cv2.imread(self.staff_dict['image path'],0)
        staff_slices = []
        start_xmin = boxes[0][1]
        right_margin = boxes[-1][3]
        right_margin = self.staff_dict['right margin']
        end_xmin = np.maximum(boxes[-1][3], right_margin)
        i=0
        while start_xmin+ i*self.sampling_frequency +self.slice_width < right_margin:
            xmin = start_xmin+i*self.sampling_frequency 
            xmax= xmin+self.slice_width
            staff_slice = [self.upper_staff_margin,xmin,self.lower_staff_margin,xmax] 
            staff_slices.append(staff_slice)
            i+=1
        staff_slices = self.correct_stave_slice_boxes_for_text(staff_slices)                
        list_of_stave_lines_per_stave_slice= []
        for staff_slice_box in staff_slices:
            staves=get_stave_line_candidates(grayscale, staff_slice_box,self.inter_staff_line_spacing)
            staves = staff_slice_box[0]+np.asarray(staves)
            list_of_stave_lines_per_stave_slice.append(list(staves)) 
        return list_of_stave_lines_per_stave_slice, staff_slices

    def get_middle_and_middle_index_within_trace(self,dict_of_middles, most_voted_key):
        middle_trace= dict_of_middles[most_voted_key]
        scoring_range = 6
        max_score= 25
        max_accumulated_score= scoring_range*max_score
        accumulated_scores = []
        i = 0
        while i <len(middle_trace)-scoring_range:
            accumulated_score=  np.sum([ el[1] for el in middle_trace[i:i+scoring_range]])
            if accumulated_score == max_accumulated_score:
                accumulated_scores.append(accumulated_score)
                break
            else:
                accumulated_scores.append(accumulated_score)
                i+=1     
        middle_index = middle_trace[np.argmax(accumulated_scores)][2]
        middle = middle_trace[middle_index][0]
        return middle, middle_index 

    def infer_staff_from_line(self,line,n_up_lines, n_down_lines, detected_lines_in_histogram):
        minimal_inter_staff_line_spacing = self.inter_staff_line_spacing-self.staff_line_identifiability_margin
        if line + n_up_lines*minimal_inter_staff_line_spacing> self.upper_staff_margin\
                                and line - n_down_lines*minimal_inter_staff_line_spacing> 0 :  
            constructed_staffs = [[line]]
            for i in range(0,n_up_lines):
                newly_constructed_staffs = []   
                for constructed_staff in constructed_staffs:
                    upwards_extended_staffs = self.extend_line_up(constructed_staff,detected_lines_in_histogram)
                    for staff in upwards_extended_staffs:
                        newly_constructed_staffs.append(staff)
                constructed_staffs = newly_constructed_staffs
            for i in range(0,n_down_lines):
                newly_constructed_staffs = []   
                for constructed_staff in constructed_staffs:
                    downwards_extended_staffs =self.extend_line_down(constructed_staff,detected_lines_in_histogram)                        
                    for staff in downwards_extended_staffs:
                        newly_constructed_staffs.append(staff)
                constructed_staffs = newly_constructed_staffs
        else: 
            constructed_staffs  = []
        return constructed_staffs

    def infer_middles_from_detected_lines_in_histogram(self,detected_lines_in_histogram):
        all_middles , all_scores = [],[]
        for line in detected_lines_in_histogram:
            middles, scores  = self.infer_all_scored_middles_from_given_line(line,detected_lines_in_histogram)
            all_middles += middles
            all_scores += scores
        if len(all_middles)>0:
            all_middles, all_scores = self.contract_middles_scores(all_middles,all_scores)
            all_scores = self.sanitize_total_middle_scores(all_scores)
        return all_middles, all_scores

    def get_constructed_middles_and_scores_from_constructed_staffs(self,constructed_staffs,
                                                                   detected_lines_in_histogram):    
        unique_middles, unique_scores = [], []
        if len (constructed_staffs) >0 :
            scores_constructed_staffs,constructed_middles = [],[]      
            for staff in constructed_staffs:
                score = self.score_staff(staff,detected_lines_in_histogram)
                if score>0:
                    scores_constructed_staffs.append(score)
                    constructed_middles.append(staff[2])
            if len(constructed_middles) >0:
                unique_middles= [constructed_middles[0]]
                unique_scores = [scores_constructed_staffs[0]]
                for middle,score in zip(constructed_middles,scores_constructed_staffs):
                    identified = False
                    for j,unique_middle in enumerate(unique_middles):
                        if np.abs(middle-unique_middle) <= self.staff_line_identifiability_margin:
                            if score > unique_scores[j] :
                                scores_constructed_staffs[j] = score
                            identified =True
                            break
                    if not identified:
                        unique_middles.append(middle)
                        unique_scores.append(score)
        return unique_middles, unique_scores  

    def extend_line_up(self,constructed_staff,detected_lines_in_histogram):
        last_line  = constructed_staff[0]
        estimated_next_line = last_line - self.inter_staff_line_spacing
        lower_bound = estimated_next_line - self.staff_line_identifiability_margin
        upper_bound = estimated_next_line + self.staff_line_identifiability_margin
        lines_up= []
        for line in detected_lines_in_histogram:
            if line >= lower_bound and line <= upper_bound:
                lines_up.append(line)
            if line > upper_bound:
                break
        if len(lines_up) ==0:
            lines_up.append(estimated_next_line)
        newly_constructed_staffs = []
        for line in lines_up:
            newly_constructed_staff = [line] + constructed_staff 
            newly_constructed_staffs.append(newly_constructed_staff)   
        return newly_constructed_staffs   

    def extend_line_down(self,constructed_staff,detected_lines_in_histogram):
        last_line  = constructed_staff[-1]
        estimated_next_line = last_line+ self.inter_staff_line_spacing
        lower_bound = estimated_next_line - self.staff_line_identifiability_margin
        upper_bound = estimated_next_line + self.staff_line_identifiability_margin
        lines_down = []
        for line in detected_lines_in_histogram:
            if line >= lower_bound and line <= upper_bound:
                lines_down.append(line)
            if line > upper_bound:
                break
        if len(lines_down) ==0:
            lines_down.append(estimated_next_line)
        newly_constructed_staffs = []
        for line in lines_down:
            newly_constructed_staff = constructed_staff + [line]
            newly_constructed_staffs.append(newly_constructed_staff)                                        
        return newly_constructed_staffs

    def contract_middles_scores(self,middles,scores):
        unique_middles= [middles[0]]
        unique_scores = [scores[0]]
        for i in range(1, len(middles)):
            middle = middles[i]
            present = False
            for j in range(0, len(unique_middles)):
                unique_middle = unique_middles[j]
                if np.abs(unique_middle-middle)<= self.staff_line_identifiability_margin:
                    unique_scores[j]+= scores[i]
                    present = True
            if not present:
                unique_middles.append(middle)
                unique_scores.append(scores[i])
        return unique_middles, unique_scores 

    def get_most_voted_key(self,dict_of_middles):
        dict_with_accumulated_scores = {}
        for middle in dict_of_middles.keys():
            total_score = 0
            for entry in dict_of_middles[middle]:
                score = entry[1]
                total_score += score
            dict_with_accumulated_scores[middle] = total_score
        max_score_pos =np.argmax(list(dict_with_accumulated_scores.values()))
        return list(dict_with_accumulated_scores)[max_score_pos]

    def score_staff(self,constructed_staff,detected_lines_in_histogram):
        score = 0  
        if constructed_staff[-1] <= self.lower_staff_margin and constructed_staff[0] >= self.upper_staff_margin:
            for line in constructed_staff:
                if line in detected_lines_in_histogram:
                    score +=1 
        return score   

    def infer_all_scored_middles_from_given_line(self,line,detected_lines_in_histogram):
        middles = []
        scores = []   
        for j in range(0,5):
            n_down_lines = 4-j
            n_up_lines = j
            constructed_staffs= self.infer_staff_from_line(line,n_up_lines, n_down_lines,
                                                           detected_lines_in_histogram,)
            unique_middles, unique_scores = self.get_constructed_middles_and_scores_from_constructed_staffs(
                                                                constructed_staffs,detected_lines_in_histogram)        
            middles +=unique_middles                                                                            
            scores += unique_scores
        if len(middles)>0:
            middles, scores = self.contract_middles_scores(middles,scores)
        return middles, scores 

    def sanitize_total_middle_scores(self,scores):
        """
        Recap of voting mechanism: a full staff i.e 5 stave line group is induced by each line in the 
        histogram, ideally, you only see 5 lines detected after binarization. Each of these lines should
        infer the correct middle 5 times i.e. max score = 25
        """
        return [np.minimum(25,score) for score in scores]

    def hor_overlap(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = np.maximum(boxA[1], boxB[1])
        xB = np.minimum(boxA[3], boxB[3])
        boxA_width = boxA[3]-boxA[1]
        # compute the area of intersection rectangle
        intersection = np.abs(np.maximum(xB - xA, 0))/boxA_width
        return intersection

    def correct_stave_slice_boxes_for_text(self,stave_slice_boxes):
        kept_text_box_frac = 0.3
        text_boxes= self.staff_dict['text boxes']
        staff_middle =  0.5*(self.upper_staff_margin+self.lower_staff_margin)
        for stave_slice_box in stave_slice_boxes:
            for text_box in text_boxes:
                if self.hor_overlap(stave_slice_box, text_box)>0:
                    #upper half
                    if text_box[0] < staff_middle:
                        stave_slice_box[0]= int(text_box[2]-kept_text_box_frac*(text_box[2]-text_box[0]))
                    #lower half
                    if  text_box[0] > staff_middle:
                        stave_slice_box[2]= int(text_box[0]+kept_text_box_frac*(text_box[2]-text_box[0]))                    
                if text_box[1] > stave_slice_box[3]:
                    break
        return stave_slice_boxes   
