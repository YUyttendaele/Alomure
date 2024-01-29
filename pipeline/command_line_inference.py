from inference import *
import sys 
sys.path.append('./dcnns/yolov7/')
sys.path.append('./post_dcnn_inference/')
from post_dcnn_inference.parse_folium import Parse_Folium
from detect import *

import pandas as pd

def write_music_data_to_df(img_path,input_folder):
    print('Parsing folium {}'.format(img_path))
    specialized_detections = True
    superclasses = True
    reference_input_folder  = './ref_symbols/'
    folium_parser=  Parse_Folium(img_path,reference_input_folder, superclasses,specialized_detections)
    list_of_staff_dicts= folium_parser.get_parsed_folium()
    print('Done Parsing folium {}'.format(img_path))    
    
    
    img_name = img_path.split('/')[-1]
    df_list= []
    for staffn, staff_dictionary in enumerate(list_of_staff_dicts):
        boxes = list_of_staff_dicts[staffn]['boxes']
        label_ids = staff_dictionary['label_ids']   
        staff_positions = staff_dictionary['staff_positions']
        for (box,label_id,staff_position) in zip(boxes,label_ids, staff_positions):
            ymin = box[0] 
            ymax = box[2] 
            xmin = box[1] 
            xmax = box[3] 
            df_list.append([label_id,staff_position,ymin,xmin,ymax,xmax])   
        column_name = ['label id','staff position', 'ymin', 'xmin', 'ymax', 'xmax']
        df = pd.DataFrame(df_list, columns=column_name)       
        output_ext_csv = img_name.split('.jp')[0]+'_'+str(staffn)+'.csv' 
        output_filename_csv = os.path.join(input_folder,'detection_dicts',img_name.split('.jp')[0],output_ext_csv) 
        df.to_csv(output_filename_csv, index=None)            
        df_list=[]   
    return  list_of_staff_dicts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the folder containing the folia, for example: "./test_omr"')
    parser.add_argument('--weights', type=str, required=False,
                        help='Path to the .pt weights file "')     

    parser.add_argument('--correction',type=str, default='N', help=' two options: Y/N . If "Y" then the predictions will be shown in Labelimg after which corrections to \
    stave postion and label symbol can be made.  ')                 
    args = parser.parse_args()

    # object detection
    aspect_ratio = 1
#    detector= object_detection_yolov7(args.input_folder,aspect_ratio)
#    detector.object_detection()
    
    
    image_paths=glob.glob(args.input_folder+'/'"*.jp*g") 
    for img_path in image_paths: 
        parsed_folium = write_music_data_to_df(img_path,args.input_folder)
