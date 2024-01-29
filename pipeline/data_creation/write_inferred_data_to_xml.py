import sys 
from ../data_reformatting.detected_data_to_pascal_voc import *
import math
def write_music_data_to_xml(img_path):
    """
    The symbols are written away in the xml format so that we can adjust them in an annotation program.
    """ 
    img_name = img_path.split('/')[-1]
    input_folder = img_path.split(img_name)[0]
    CONFIDENCE_THRESHOLD = 0.15
    OVERLAP_THRESHOLD = 0.4
    clean_dict=  Duration_Dictionaries(img_path, CONFIDENCE_THRESHOLD , OVERLAP_THRESHOLD)   
    list_of_dict= clean_dict.get_corrected_boxes_and_classes()
    list_of_dict=rescale_dictionaries(list_of_dict,scale_factor_width,scale_factor_height)
    for staffn, staff_dictionary in enumerate(list_of_dict):
        staff_dictionary= list_of_dict[staffn]
        positions_on_staff_finder= Positions_on_staff(img_path,staff_dictionary)
        staff_dictionary = positions_on_staff_finder.get_dictionary_with_positions()
        boxes = staff_dictionary['boxes']
        classes = staff_dictionary['classes']
        labels = [inv_label_dict[class_id] for class_id in classes]
        margins = [staff_dictionary['ymin'], staff_dictionary['ymin']+ staff_dictionary['staff_height']]
        positions_on_staff = staff_dictionary['staff_positions']     
        composed_labels = []
        for box, label,position_on_staff in zip(boxes,labels,positions_on_staff):
            if type(position_on_staff) == str and len(position_on_staff) !=0:                
                composed_labels.append(label+'_'+str(position_on_staff))          
            else:
                composed_labels.append(label)        
        target_image_name = img_name.split('.jp')[0]+'_'+str(staffn)+'.'+img_path.split('.')[-1]
        output_ext_xml = img_name.split('.jp')[0]+'_'+str(staffn)+'.xml' 
        target_image_path = os.path.join(input_folder,'detections',img_name.split('.jp')[0], target_image_name )
        shutil.copyfile(img_path, target_image_path)    
        output_filename_xml = os.path.join(input_folder,'detections',img_name.split('.jp')[0],output_ext_xml) 
        savePascalVocFormat(output_filename_xml,boxes,composed_labels, target_image_path, lineColor=None, fillColor=None, databaseSrc=None) 
