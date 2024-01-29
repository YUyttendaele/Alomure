import numpy as np
import sys
sys.path.append('../data_creation/')
from get_label_dict import get_dict_of_label_ids
label_dict= get_dict_of_label_ids()
inv_label_dict = {v: k for k, v in label_dict.items()}

notes_labels = ['maxima', 'longa', 'colored longa', 'breve', 'colored breve', 'semibreve', 'colored semibreve',
         'minim', 'semiminim', 'fusa', 'semifusa', 'l1longa', 'l2longa', 'l1colored longa', 'l2colored longa',
         'l1breve', 'l2breve', 'l1colored breve', 'l2colored breve', 'l1semibreve', 'l2semibreve',
         'l1colored semibreve', 'l2colored semibreve', 'o1longa', 'o1colored longa', 'o2longa',
         'o2colored longa', 'o1breve', 'o1colored breve', 'o2breve', 'o2colored breve', 'o1semibreve',
         'o1colored semibreve', 'o2semibreve', 'o2colored semibreve']
notes_ids = [label_dict[label] for label in notes_labels]    

symbols_appearing_on_top_of_other_labels = ['dot','fermata','congruence','flat',\
                                           'imaj', 'pmaj', 'imin', 'imincut', 'pmincut',
                                            '3', '2', '1']
symbols_appearing_on_top_of_other_labels = ['dot','fermata','congruence','flat']

symbols_appearing_on_top_of_other_ids  = [label_dict[label] for label in symbols_appearing_on_top_of_other_labels]

ligature_labels = [ 'l1longa', 'l2longa', 'l1colored longa', 'l2colored longa','l1breve',
                    'l2breve', 'l1colored breve','l2colored breve', 'l1semibreve', 'l2semibreve',
                   'l1colored semibreve', 'l2colored semibreve', 'o1longa','o1colored longa', 'o2longa',             
                    'o2colored longa', 'o1breve', 'o1colored breve', 'o2breve', 'o2colored breve', 
                   'o1semibreve', 'o1colored semibreve', 'o2semibreve', 'o2colored semibreve',
                   'l1','l2','o1','o2','colored l1', 'colored o1', 'colored l2', 'colored o2']
ligature_ids = [label_dict[label] for label in ligature_labels]    

class Overlapping_Box_Filter2(object):         
    def remove_boxes_overlapping_with_text(self, staff_dict):
        allowed_overlap =  0.5       
        kept_indices = []
        for i,box in enumerate(staff_dict['boxes']):
            discard = False
            for text_box in staff_dict['text boxes']:
                if self.box_overlap(box,text_box)>allowed_overlap:
                    discard = True
                    break
            if not discard:
                kept_indices.append(i)
        #remove text boxes overlapping with text boxex                
        for key in ['scores','boxes','label_ids']:
            staff_dict[key] = list(np.asarray(staff_dict[key])[kept_indices])
        staff_dict= self.sort_boxes_left_to_right(staff_dict) 
        discarded_text_boxes = []
        allowed_overlap_text = 0.2
        for i in range(len(staff_dict['text boxes'])-1):
            boxi= staff_dict['text boxes'][i]
            for j in range(i+1,len(staff_dict['text boxes'])):
                boxj= staff_dict['text boxes'][j]
                if self.box_overlap(boxi,boxj)>allowed_overlap_text or self.box_overlap(boxj,boxi)>allowed_overlap_text :
                    #if (boxi[2]-boxi[0]) < boxj[2]-boxj[0] :
                        discarded_text_boxes.append(i)
                        break
                    
        kept_text = [index for index in range(len(staff_dict['text boxes'])) if index not in discarded_text_boxes]
        staff_dict['text boxes'] = list(np.asarray(staff_dict['text boxes'])[kept_text] )           
        return staff_dict
    
    
    def remove_overlapping_bounding_boxes_from_folium(self,staff_dictionaries):
        overlap_threshold = 0.5
        confidence_threshold = 0.17
        filtered_folium = []
        for staffnumber,staff_dict in enumerate(staff_dictionaries):
            staff_dict = self.remove_boxes_overlapping_with_text(staff_dict)
            #get highly overlapping indices (point,fermata,congruence )
            symbols_appearing_on_top_of_other_indices = [index for index in range(len(staff_dict['label_ids'])) 
                                           if staff_dict['label_ids'][index] in symbols_appearing_on_top_of_other_ids]
            filtered_symbols_appearing_on_top_of_other_indices  = self.mensural_suppression(staff_dict,\
                        symbols_appearing_on_top_of_other_indices,overlap_threshold,'symbols appearing on top of other',\
                                                                                confidence_threshold,staffnumber)
#             #get ligature indices
            ligature_indices = [index for index in range(len(staff_dict['label_ids'])) 
                                          if staff_dict['label_ids'][index] in ligature_ids]
            filtered_ligature_indices = self.mensural_suppression(staff_dict, ligature_indices,\
                                                        overlap_threshold,'ligatures',confidence_threshold,staffnumber) 

            # get other indices
            other_indices = [index for index in range(len(staff_dict['label_ids'])) if not\
                             (index in symbols_appearing_on_top_of_other_indices or index in ligature_indices)]
            filtered_other_indices =  self.mensural_suppression(staff_dict, other_indices, overlap_threshold,\
                                                                   'other',confidence_threshold,staffnumber)            

            # add indices 
            filtered_indices =self.fuse_boxes(staff_dict,filtered_symbols_appearing_on_top_of_other_indices,\
                                                filtered_ligature_indices,filtered_other_indices,overlap_threshold)
            for key in ['scores','boxes','label_ids']:
                staff_dict[key] = list(np.asarray(staff_dict[key])[filtered_indices])
            staff_dict= self.sort_boxes_left_to_right(staff_dict) 
 
            new_xmin = []
            new_xmax = []     
            com_vert = 0.25

            for i in range(len(staff_dict['boxes'])-1):
                xmin_next = staff_dict['boxes'][i+1][1]  
                xmax_next = staff_dict['boxes'][i+1][3]  
                ymin_next = staff_dict['boxes'][i+1][0]   
                ymax_next = staff_dict['boxes'][i+1][2]             
                boxA = [ymin_next, xmin_next, ymax_next, xmax_next]
                
                xmax_current =  staff_dict['boxes'][i][1]
                xmin_current =  staff_dict['boxes'][i][3]
                ymax_current =  staff_dict['boxes'][i][0]
                ymin_current =  staff_dict['boxes'][i][2]               
                boxB = [ymin_current, xmin_current, ymax_current, xmax_current]
                

                if xmin_next < xmax_current  and sufficient_ver_box_overlap(boxA, boxB,com_vert)\
                and sufficient_ver_box_overlap(boxB, boxA,com_vert) and  xmin_next>xmin_current:
                    staff_dict['boxes'][3][1] = int(0.5* (xmax_current+xmin_next))                     
                    staff_dict['boxes'][i+1][1]  = int(0.5* (xmax_current+xmin_next)) 
            
            if len(staff_dict['boxes'])>2:
                filtered_folium.append(staff_dict)
       
        return filtered_folium

    

    def sufficient_ver_box_overlap(boxA, boxB,minimum_common_length_fraction):
        # determine the (x, y)-coordinates of the intersection rectangle

        xA = np.maximum(boxA[0], boxB[0])
        xB = np.minimum(boxA[2], boxB[2])

        boxA_length = boxA[2]-boxA[0]
        # compute the area of intersection rectangle
        intersection = np.abs(np.maximum(xB - xA, 0))
        return intersection>=minimum_common_length_fraction*boxA_length       
       
    
    
    
    
    
    def sort_boxes_left_to_right(self, staff_dict): 
        xmins = [box[1] for box in staff_dict["boxes"]]
        sorted_idxs = np.argsort(xmins)  
        for key in ['scores','label_ids','boxes']:
            staff_dict[key]= np.asarray(staff_dict[key])[sorted_idxs]
        return staff_dict   
            
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

    def hor_overlap(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = np.maximum(boxA[1], boxB[1])
        xB = np.minimum(boxA[3], boxB[3])

        boxA_width = boxA[3]-boxA[1]
        # compute the area of intersection rectangle
        intersection = np.abs(np.maximum(xB - xA, 0))/boxA_width
        return intersection

    def compatible_ligature_ids(self,lig_id1,lig_id2):
        label_lig_id1 = inv_label_dict[lig_id1]
        label_lig_id2 = inv_label_dict[lig_id2]
        return ('l1' in label_lig_id1 and 'l2' in label_lig_id2) 

    def overlap_criterion(self,overlap_mode,staff_dict,indexA,indexB,discarded_indices,overlap_threshold):
        if overlap_mode in ['ligatures', 'other']:        
            return indexB not in discarded_indices and\
            (self.box_overlap(staff_dict['boxes'][indexB],staff_dict['boxes'][indexA]) > overlap_threshold or\
            self.box_overlap(staff_dict['boxes'][indexA],staff_dict['boxes'][indexB]) >  overlap_threshold ) and\
            not self.compatible_ligature_ids(staff_dict['label_ids'][indexA],staff_dict['label_ids'][indexB])
        if overlap_mode == 'symbols appearing on top of other':
            return indexB not in discarded_indices and\
            (self.box_overlap(staff_dict['boxes'][indexB],staff_dict['boxes'][indexA]) >  overlap_threshold  or\
            self.box_overlap(staff_dict['boxes'][indexA],staff_dict['boxes'][indexB]) >  overlap_threshold )         

    def fuse_boxes(self,staff_dict,filtered_symbols_appearing_on_top_of_other_indices,filtered_ligature_indices,filtered_other_indices,overlap_threshold): 
        discarded_other_indices = []
        discarded_on_top_indices = []
        for index in filtered_other_indices:
            overlap_mode = 'other'
            for lig_index in filtered_ligature_indices:
                if self.overlap_criterion(overlap_mode,staff_dict,index,lig_index,discarded_other_indices,overlap_threshold):
                    discarded_other_indices.append(index)
            overlap_mode = 'symbols appearing on top of other'
            for on_top_index in filtered_symbols_appearing_on_top_of_other_indices:
                if inv_label_dict[staff_dict['label_ids'][on_top_index]] not in ['dot','fermata','congruence'] and\
                    self.overlap_criterion(overlap_mode,staff_dict,index,on_top_index,discarded_on_top_indices,
                                            overlap_threshold):
                        if  staff_dict['scores'][on_top_index] > staff_dict['scores'][index]: 
                            discarded_other_indices.append(index)

                        else:
                            discarded_on_top_indices.append(on_top_index)
                elif inv_label_dict[staff_dict['label_ids'][on_top_index]] in ['dot','fermata','congruence'] and\
                    staff_dict['label_ids'][index] not in notes_ids and\
                    self.overlap_criterion(overlap_mode,staff_dict,index,on_top_index,discarded_on_top_indices,\
                                            overlap_threshold):
                        if  staff_dict['scores'][on_top_index] > staff_dict['scores'][index]:
                            discarded_other_indices.append(index)
                        else:
                            discarded_on_top_indices.append(on_top_index)                        
                    
        kept_other_indices = [index for index in filtered_other_indices if index not in discarded_other_indices]
        kept_on_top_indices = [index for index in filtered_symbols_appearing_on_top_of_other_indices\
                               if index not in discarded_on_top_indices]
        
        return kept_other_indices+filtered_ligature_indices+kept_on_top_indices
    
    def mensural_suppression(self,staff_dict, indices, threshold,overlap_mode,confidence_threshold, staff_number):
        if len(indices) !=0:
            indices = [index for index in indices if staff_dict['scores'][index] > confidence_threshold]
            discarded_indices = []
            most_probable_indices = []
            most_probable_boxes= []
            for i in range(len(indices)):
                if indices[i] not in discarded_indices:
                    j=i+1
                    #collect all overlapping 
                    overlapping_indices_and_scores= [[indices[i],staff_dict['scores'][indices[i]]]]
                    while j < len(indices) and self.box_overlap(staff_dict['boxes'][indices[j]],staff_dict['boxes'][indices[i]])>0:
                        if self.overlap_criterion(overlap_mode,staff_dict,indices[i],indices[j],discarded_indices,threshold):
                            overlapping_indices_and_scores.append([indices[j], staff_dict['scores'][indices[j]]])
                        j+=1
                    scores=  [el[1] for el in  overlapping_indices_and_scores]
                    overlapping_indices=  [el[0] for el in  overlapping_indices_and_scores]
                    sorted_idxs = np.argsort(scores)[::-1]  
                    discarded_indices += list(np.asarray(overlapping_indices)[sorted_idxs[1:]])                    
                    most_probable_index = np.asarray(overlapping_indices)[sorted_idxs[0]]
                    if most_probable_index not in most_probable_indices:
                        most_probable_indices.append(most_probable_index)  
            if overlap_mode == 'ligatures':
                filtered_indices=[]
                i =0 
                while i < len(most_probable_indices):
                    chain = [most_probable_indices[i]]
                    if i < len(most_probable_indices)-1:
                        j = i+1
                        while j< len(most_probable_indices)-1 and (self.hor_overlap(staff_dict['boxes'][chain[-1]],staff_dict['boxes'][most_probable_indices[j]]) > threshold or\
                                self.hor_overlap(staff_dict['boxes'][most_probable_indices[j]],staff_dict['boxes'][chain[-1]]) > threshold):
                                chain.append(most_probable_indices[j])
                                j+=1
                    filtered_indices += [chain[len(chain)-1-k] for k in range(0,len(chain),2)]
                    if len(chain)>1 and self.compatible_ligature_ids(staff_dict['label_ids'][chain[0]],\
                                staff_dict['label_ids'][chain[1]]) and chain[0] not in filtered_indices:
                        filtered_indices.append(chain[0])    
                    i+= len(chain)
                return filtered_indices
            else: 
                return most_probable_indices
        else:
            return indices
