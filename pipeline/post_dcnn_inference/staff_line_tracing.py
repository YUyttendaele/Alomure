import numpy as np
from staff_middle_line_finder import *

class Folium_Staff_Line_Tracing(object):       
    def add_staff_lines_to_folium_music_symbols(self,folium):
        for staff_dict in folium:
            middle_finder = Middle_Finder(staff_dict)
            sample_frequency = int((1/3)*staff_dict['inter_staff_line_spacing'])
            dict_of_middles,middle_line,staff_box_index= middle_finder.get_middle()
            list_of_staff_lines_per_staff_slice = middle_finder.get_list_of_staff_lines_per_staff_slice()  
            staff_dict= self.get_dictionaries_with_reconstructed_staffs(staff_dict, middle_line,
                                    staff_box_index,list_of_staff_lines_per_staff_slice,sample_frequency)                                                         
        return folium
    
    def get_dictionaries_with_reconstructed_staffs(self,staff_dictionary, middle_line,staff_box_index,
                                                   list_of_stave_lines_per_stave_slice,sample_frequency ):
        boxes = staff_dictionary['boxes']
        inter_staff_line_spacing = staff_dictionary['inter_staff_line_spacing']
        staff_reconstructer =Staff_Reconstructer(list_of_stave_lines_per_stave_slice,inter_staff_line_spacing,
                                                 middle_line, staff_box_index,boxes,sample_frequency)
        staff_lines_around_music_symbols= staff_reconstructer.get_inferred_staff_lines_around_music_symbols()
        staff_dictionary['staff_lines_around_music_symbol'] = staff_lines_around_music_symbols 
        return staff_dictionary 

class Staff_Reconstructer(object):   

    def __init__(self,list_of_stave_lines_per_stave_slice,inter_staff_line_spacing,middle_line,
                 staff_box_index,boxes,sample_frequency):   
        self.inter_staff_line_spacing = inter_staff_line_spacing
        self.list_of_stave_lines_per_stave_slice= list_of_stave_lines_per_stave_slice
        self.start_middle = middle_line
        self.boxes = boxes
        self.start_index = staff_box_index
        
        self.sample_frequency = sample_frequency
        #hyperparameters
        self.staff_line_identification_bound = int(np.round(0.35*self.inter_staff_line_spacing))        
        self.n_look_aheads = 5

    def smoothen_staff(self,constructed_staff):
        """
        As a consequence of faulty detections, specific stave lines might diverge
        from the near equidistant 5-line pattern and error propagation might occur when line tracking a 
        stave line over the staff width. To mitigate aforementioned phenomenon, we implement the current method which votes on
        the inferred middle and builds the other stave lines around it. To within pre-specified bounds, deviations of a stave line from its expected height 
        within the 5-stave line patern are allowed.
        """
        
        # each line produces an estimate of the middle
        middle_estimates = []
        for index, line in enumerate(constructed_staff): 
            estimate = line + (2-index) *self.inter_staff_line_spacing
            middle_estimates.append(estimate)
        
        #remove outliers 
        sorted_middle_estimates= np.sort(middle_estimates)
        outlier_removed_middle_estimates = sorted_middle_estimates
        mean_middle_estimate =int(np.ceil(np.mean(outlier_removed_middle_estimates)))   
        
        #build smoothened staff
        evidence = constructed_staff
        middle = mean_middle_estimate
        line_1 = self.get_closest_line(middle-self.inter_staff_line_spacing,evidence )
        line_0 = self.get_closest_line(line_1-self.inter_staff_line_spacing,evidence )
        line_3 = self.get_closest_line(middle+self.inter_staff_line_spacing,evidence )
        line_4 = self.get_closest_line(line_3+self.inter_staff_line_spacing,evidence )
        
        return [line_0,line_1, middle, line_3, line_4]

        
    def get_closest_line(self,line, evidence):
        closest_line = evidence[0]
        for evidence_line in evidence:
            if np.abs(evidence_line-line)<np.abs(line-closest_line):
                closest_line = evidence_line
        
        if np.abs(line -closest_line) <= self.staff_line_identification_bound/2:
            return closest_line
        else:
            return line
                      
    def construct_initial_middle(self):
        middle= self.start_middle
        evidence = self.list_of_stave_lines_per_stave_slice[self.start_index]
        line_1 = self.get_closest_line(middle-self.inter_staff_line_spacing,evidence )
        line_0 = self.get_closest_line(line_1-self.inter_staff_line_spacing,evidence )
        line_3 = self.get_closest_line(middle+self.inter_staff_line_spacing,evidence )
        line_4 = self.get_closest_line(line_3+self.inter_staff_line_spacing,evidence )
        return [line_0, line_1,middle,line_3,line_4]
    
    def reconstruct_staff(self,index,previous_staff, update_direction):
        sorted_surrounding_staffs = self.get_n_surrounding_elements(index,\
                                                self.list_of_stave_lines_per_stave_slice, self.n_look_aheads)
        reconstructed_staff = []
        for line_number,previous_line in enumerate(previous_staff):
            reconstructed_staff.append(previous_line)
            candidate_lines=[]
            for index,staff in enumerate(sorted_surrounding_staffs):        
                if len(staff)!=0:
                    best_approx_stave_line =  [previous_staff[2]+(index-2)*self.inter_staff_line_spacing]
                    minimal_distance  = np.abs(previous_line-best_approx_stave_line[0])
                    for stave_line in staff:
                        distance = np.abs(stave_line - previous_line)
                        if distance <= minimal_distance:
                            minimal_distance = distance
                            best_approx_stave_line= [stave_line]
                        if distance == minimal_distance:
                            minimal_distance = distance
                            best_approx_stave_line.append(stave_line)
                    if minimal_distance <= self.staff_line_identification_bound:
                        candidate_lines.append([best_approx_stave_line,index])  
            if len(candidate_lines)!= 0:
                reconstructed_staff[-1]=self.update_line(previous_line,candidate_lines)    
        reconstructed_staff=self.smoothen_staff(reconstructed_staff)
        return reconstructed_staff      
    
    def get_inferred_staff_lines_around_music_symbols(self):
        all_inferred_staff_lines = self.get_inferred_staff_lines()
        staff_lines_around_music_symbols = []
        box_middles = [0.5*(box[1]+box[3]) for box in self.boxes]
        for box_middle in box_middles:
            staff_line_box_index = np.minimum(int((box_middle-self.boxes[0][1])/self.sample_frequency),\
                                             len(all_inferred_staff_lines)-1)
            staff_lines_around_music_symbols.append(all_inferred_staff_lines[staff_line_box_index])
        return staff_lines_around_music_symbols 
    

    def update_line(self,previous_line,candidate_lines):
        gradients = []
        for el in candidate_lines:
            lines, index = el[0],el[1]
            for line in lines:
                gradient = ((line- previous_line)/((1+index)*self.sample_frequency))
                gradients.append(gradient)  
        average_gradient = np.mean(gradients)
        increment = self.sample_frequency*average_gradient
        return previous_line + np.sign(increment)*np.round(np.abs(increment))
    
    def get_inferred_staff_lines(self):    
        start_staff = self.construct_initial_middle()
        previously_constructed_staff =start_staff
        reverse_staffs = []
        for i in range(0, self.start_index+1):
            correct_staff=self.reconstruct_staff(self.start_index-i,previously_constructed_staff,update_direction=-1 ) 
            previously_constructed_staff  = correct_staff 
            full_staff = self.construct_full_staff(correct_staff)
            reverse_staffs.append(full_staff)      
        self.correct_staffs= self.flip_list(reverse_staffs)
        previously_constructed_staff =start_staff
  
        for i in  range(self.start_index+1,len(self.list_of_stave_lines_per_stave_slice)):
            correct_staff=self.reconstruct_staff(i,previously_constructed_staff,update_direction=1 ) 
            previously_constructed_staff  = correct_staff 
            full_staff = self.construct_full_staff(correct_staff)
            self.correct_staffs.append(full_staff)         
        return self.correct_staffs
    
    def construct_full_staff(self,staff):
        full_staff = [
        staff[0]- int(np.round(1.5*self.inter_staff_line_spacing)),
        staff[0]- self.inter_staff_line_spacing,
        staff[0]- int(np.round(0.5*self.inter_staff_line_spacing)),
        staff[0],
        staff[1]- int(np.round(0.5*self.inter_staff_line_spacing)),
        staff[1],
        staff[2]- int(np.round(0.5*self.inter_staff_line_spacing)),
        staff[2],
        staff[3]- int(np.round(0.5*self.inter_staff_line_spacing)),
        staff[3],
        staff[4]- int(np.round(0.5*self.inter_staff_line_spacing)),
        staff[4],
        staff[4]+ int(np.round(0.5*self.inter_staff_line_spacing)),
        staff[4]+ self.inter_staff_line_spacing,
        staff[4]+ int(np.round(1.5*self.inter_staff_line_spacing))
        ]
        return full_staff   

    def get_n_surrounding_elements(self,index, list, n):
        # n = odd and smaller than len(list)
        margin = int(0.5*(n-1))
        add_right= 0
        add_left= 0
        if index-margin >= 0: 
            add_right =0
        else: 
            add_right= margin-index

        if len(list)-1-(margin +index) >=0:
            add_left = 0
        else:
            add_left = margin+index-len(list)+1
        lowerbound = np.maximum(0,index-(margin+add_left))
        upperbound = np.minimum(len(list)-1,index+margin+add_right)
        sorted_list = [list[index]]
        i =1
        within_bounds = True
        while i< n and within_bounds:
            within_bounds= False
            if index+i <= upperbound:
                sorted_list.append(list[index+i])  
                within_bounds = True

            if index-i >= lowerbound:
                sorted_list.append(list[index-i])
                within_bounds = True
            i+=1         
        return sorted_list        

    def flip_list(self,list):
        flipped = []
        for i in range(len(list)):
            flipped.append(list[len(list)-1-i])
        return flipped  
