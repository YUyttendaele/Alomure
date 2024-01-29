import cv2 
import sys

sys.path.append('./data_creation/')
from get_label_dict import get_dict_of_label_ids

sys.path.append('./post_dcnn_inference/')
from filter_staffs_load_music_symbols import *
from overlapping_box_filtering import *
from inter_staff_line_spacing_finder import *
from staff_line_tracing import *
from get_positions_on_staff import *
from super_class_parsing import *
from detection_error_correction import *

class Parse_Folium(object):
    """"
    Returns a list of sorted dictionaries: one for every staff, each staf dictionary contains the inferred notes
    and their respective positions.
    """    
    def __init__(self,img_path, ref_database_input_folder,superclasses,specialized_dections):     
        #load dictionary
        self.img_path=img_path   
        self.input_folder = self.img_path.split(self.img_path.split('/')[-1])[0]
        self.grayscale=cv2.imread(self.img_path,0)
        self.ref_database =ref_database_input_folder
        self.superclasses = superclasses
        self.specialized_dections = specialized_dections

    def get_parsed_folium(self):        
        #Load, filter and sort staffs , load --unfiltered-- music symbols
        staff_filter_and_music_symbol_loader=  Staff_Filtering_and_Music_Symbol_Loading(self.input_folder,\
                    self.img_path,self.specialized_dections,scale_factor_width = None,scale_factor_height= None)        
        list_of_staff_dictionaries= staff_filter_and_music_symbol_loader.get_filtered_staffs_with_music_symbols()
        #Filter overlapping music symbols
        overlapping_music_symbol_filter = Overlapping_Box_Filter2()
        list_of_staff_dictionaries =\
        overlapping_music_symbol_filter.remove_overlapping_bounding_boxes_from_folium(list_of_staff_dictionaries)
#         #Add inter staff line spacing for every staff 
        inter_staff_spacing_finder = Inter_Staff_Line_Spacing_Finder(list_of_staff_dictionaries,self.grayscale)
        list_of_staff_dictionaries = inter_staff_spacing_finder.add_inter_staff_spacing_estimated_to_dicts() 

        #Reconstructed staffs: for every staff in the dictionary we trace each stave lin
        staff_line_tracer = Folium_Staff_Line_Tracing()      
        list_of_staff_dictionaries = staff_line_tracer.add_staff_lines_to_folium_music_symbols(\
                                                                    list_of_staff_dictionaries)         
        #Add position of symbol on staff 
        Position_on_staff_finder = Positions_on_staff(self.img_path)
        list_of_staff_dictionaries =Position_on_staff_finder.attribute_position_on_staff_to_music_symbols(list_of_staff_dictionaries)        
#         #If the dcnn contains super classes, parse them:
        if self.superclasses:
            super_class_parsing = super_class_parser(self.ref_database)
            list_of_staff_dictionaries = super_class_parsing.get_dictionaries_with_superclasses_parsed(list_of_staff_dictionaries)                                    
        #Apply heuristics to correct the symbol detection output of the previous methods
        folium_corrector = clean_folium_heuristics(list_of_staff_dictionaries)
        list_of_staff_dictionaries  = folium_corrector.correct_folium()
        return list_of_staff_dictionaries
