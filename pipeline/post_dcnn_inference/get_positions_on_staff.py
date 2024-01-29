import numpy as np
import sys
import cv2

sys.path.append('./post_dcnn_inference/')
from pitch_nn import *
sys.path.append('./data_creation/')
from get_label_dict import get_dict_of_label_ids

label_dict= get_dict_of_label_ids()
inv_label_dict = {v: k for k, v in label_dict.items()}

symbols_with_pitch= ['maxima', 'colored maxima','longa','colored longa', 'breve',
             'colored breve','semibreve','colored semibreve','minim',
             'semiminim', 'fusa','semifusa', 'l1longa', 'l2longa',
             'l1colored longa', 'l2colored longa','l1breve','l2breve',
             'l1colored breve', 'l2colored breve', 'l1semibreve','l2semibreve',
             'l1colored semibreve', 'l2colored semibreve', 'o1longa',
             'o1colored longa', 'o2longa', 'o2colored longa', 'o1breve',
             'o1colored breve', 'o2breve', 'o2colored breve', 'o1semibreve',
             'o1colored semibreve', 'o2semibreve','o2colored semibreve', 'flat', 'custos',\
             'l1', 'l2', 'o1', 'o2', 'colored l1', 'colored o1', 'colored l2', 'colored o2']

class Positions_on_staff(object):
    def __init__(self,img_path):      
        self.staff_position_parser= staff_position_nn()
        self.img_path = img_path
        
    def attribute_position_on_staff_to_music_symbols(self, list_of_staff_dictionaries):  
        for staff_dictionary in list_of_staff_dictionaries:
            staff_positions = []        
            staff_position_scores  = []
            inter_staff_line_spacing = staff_dictionary['inter_staff_line_spacing']
            labels = [inv_label_dict[class_id] for class_id in staff_dictionary['label_ids']]
            for box,label,staff_lines in zip(staff_dictionary['boxes'],\
                    labels ,staff_dictionary['staff_lines_around_music_symbol']):
                if label in symbols_with_pitch:                
                    draw_staff_lines= False
                    pitch_img=self.get_cleaned_staff_exerpt_around_img(self.img_path ,box, staff_lines, inter_staff_line_spacing,draw_staff_lines)
                    staff_position = self.staff_position_parser.predict_staff_position(pitch_img)
                elif label in ['c clef','f clef']:
                    draw_staff_lines= False
                    pitch_img =self.get_cleaned_staff_exerpt_around_img(self.img_path ,box, staff_lines, inter_staff_line_spacing,draw_staff_lines)
                    staff_position = self.staff_position_parser.predict_clef_position(pitch_img)
                    
                elif label == 'g clef':
                    staff_position = 9
                else:
                    staff_position =None        
                staff_positions.append(staff_position)
            staff_dictionary['staff_positions'] = staff_positions
        return list_of_staff_dictionaries
   
    def get_cleaned_staff_exerpt_around_img(self,img_path,symbol_box, staff_lines,inter_staff_line_spacing,draw_staff_lines):  
        """
        cleaned excerpt has widh of symbol box, heights of the full staff i.e. 15  positions
        sets pixels outside of the symbol to white and finally normalizes by drawing the standard 5 staff lines
        """
        ymin = int(staff_lines[0]-int(0.5*inter_staff_line_spacing))
        ymax = int(staff_lines[-1]+int(0.5*inter_staff_line_spacing))        
        symbol_box_xmin, symbol_box_xmax = symbol_box[1], symbol_box[3]
        grayscale = cv2.imread(img_path)

        image_height,image_width = grayscale.shape[0],grayscale.shape[1]
        image = grayscale[ymin:ymax, symbol_box_xmin:symbol_box_xmax]
        
        #pad when full staff out of folium margin
        pixels_above_added = 0
        if ymin<0:
            pixels_above_added = -ymin            
            upper_padding = np.zeros((-ymin,image.shape[1]))
            for i in range(upper_padding.shape[0]):
                for j in range(upper_padding.shape[1]):
                    upper_padding[i][j]=255
            image = np.append(upper_padding, image, axis=0)
        if ymax > image_height:
            lower_padding = np.zeros((ymax-image_height,image.shape[1]))
            for i in range(lower_padding.shape[0]):
                for j in range(lower_padding.shape[1]):
                    lower_padding[i][j]=255        
            image = np.append(image,lower_padding,axis=0)


        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if i < symbol_box[0] -ymin or i > symbol_box[2]-ymin:
                    image[i][j] = 255  
         # find new boy ymin and ymax
        box_ymin_in_crop = symbol_box[0] - ymin + pixels_above_added   
        box_ymax_in_crop = symbol_box[2] - ymin + pixels_above_added 

        # remove noise 
#       image =self.symbol_margins_to_white(image, upper_margin = box_ymin_in_crop , lower_margin = box_ymax_in_crop )
#       if draw_staff_lines:
#            image = self.draw_staff_lines(image,staff_lines)
        
        image = image.astype('uint8')
        return image
    
    def draw_staff_lines(self,img, constructed_staff):    
        inter_staff_spacing = constructed_staff[2]-constructed_staff[0]
        ymin = constructed_staff[0]-int(0.5*inter_staff_spacing)      

        reduced_staff= [constructed_staff[i] for i in [3,5,7,9,11]]     
        reduced_staff = np.asarray(reduced_staff) - ymin

        for i in range(img.shape[0]):
            if i in reduced_staff:
                for j in range(img.shape[1]):
                    img[i][j] = 0    
        return img

    def symbol_margins_to_white(self,img, upper_margin, lower_margin):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if i < upper_margin or i > lower_margin:
                    img[i][j] = 255
        return img 
