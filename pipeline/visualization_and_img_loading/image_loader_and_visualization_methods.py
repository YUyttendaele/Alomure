import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import sys
sys.path.append('../scripts_data_reformatting/')
sys.path.append('../scripts_data_creation/')
from get_label_dict import get_dict_of_label_ids
label_dict= get_dict_of_label_ids()
inv_label_dict = {v: k for k, v in label_dict.items()}

color_dictionary= { 'minim': 'black', 'semibreve': 'crimson', 'obliqa': 'bisque', 'semiminim': 'violet',  
                   'dot': 'fuchsia', 'dotted note': 'brown', 'rest': 'lime',     'breve': 'navy', 
                   'barline': 'yellow', 'repeat': 'slategray',  'fermata': 'yellow', 'flat': 'blue', 
                   'ligature':  'teal',    'lnote1': 'chocolate', 'lnote2': 'lavenderblush', 
                    'fusa': 'darkorange', 'longa': 'slateblue','colored breve' : 'cyan', 
                   'colored semibreve': 'aquamarine', 'colored longa' : 'blue',  'c clef': 'olivedrab',
                   'f clef': 'mintcream',  'g clef': 'lightcoral',  'imaj': 'darkorange',
                   'pmaj': 'darkkhaki', 'imincut': 'darkgreen', 'imincut': 'chocolate', 'imin': 'mediumvioletred',
                   'pmin': 'saddlebrown', '3': 'mediumpurple',  '2': 'aliceblue', 'sharp': 'gold',
                   'fusa rest': 'cyan', 'semifusa':'aquamarine', '1': 'brown','custos' : 'violet',
                   'minim rest': 'teal', 'breve rest': 'pink', 'semibreve rest': 'yellowgreen',
                   'longa rest': 'slateblue', 'colored lnote1' :'green', 'colored lnote2': 'cyan',
                   'colored obliqa': 'yellow', 'white_black obliqa': 'purple', 'black_white obliqa':
                   'violet', 'pmincut': 'gray','l2colored breve': 'red','l1semibreve': 'chocolate' ,
                  'l2semibreve': 'lavenderblush', 'l1breve': 'forestgreen', 'l2breve': 'gold',
                   'congruence': 'brown', 'o1semibreve': 'red', 'o2semibreve':'red', 'o2breve' :'navy',
                   'l1colored semibreve' : 'aquamarine', 'o1colored semibreve': 'aquamarine',
                   'o2colored semibreve' : 'aquamarine',
                  'l2colored semibreve' : 'aquamarine'} 
                               
                   
def rescale_image_predefined_scale_factor(image_path):
	img = cv2.imread(image_path) 
	height, width= img.shape[0], img.shape[1]
	if width <= height: 
	    new_width = 1024
	else:
	    new_width = 2000
	scale = height/width
	img=cv2.resize(img, (new_width,int(scale*new_width)),cv2.INTER_NEAREST ) 
	scale_factor = new_width/width
	return scale_factor,img
	
def rescale_image_gray_predefined_scale_factor(image_path):
	img = cv2.imread(image_path,0) 
	height, width= img.shape[0], img.shape[1]
	if width <= height: 
	    new_width = 1024
	else:
	    new_width = 2000
	scale = height/width
	img=cv2.resize(img, (new_width,int(scale*new_width)),interpolation=cv2.INTER_NEAREST ) 
	return img        
	
def rescale_image_mpimg_predefined_scale_factor(image_path):
    img = mpimg.imread(image_path) 
    height, width= img.shape[0], img.shape[1]
    if width <= height: 
        new_width = 1024
    else:
        new_width = 2000
    scale = height/width
    img=cv2.resize(img, (new_width,int(scale*new_width)),interpolation=cv2.INTER_NEAREST ) 
    return img

def visualize_dictionary(img,save_name, list_of_dictionaries, plot_staff_margins, staffn):
    
    image=rescale_image3(img)
    fig = plt.figure(None, (10, 10), 200)
    fig.add_subplot(111)
    axes = fig.axes[0]
    axes.imshow(image,cmap="gray")
#     plt.axis('off')
    color_scheme = ['r','g','b','orange', 'cyan', 'white','brown','navy', 'fuchsia', 
    'yellow','crimson', 'black', 'bisque', 'yellow' ,'brown', 'black' ]
    for k in range(0, len(list_of_dictionaries)):
        if staffn != None :
            if staffn ==k:
                staff_ymin = list_of_dictionaries[k]["ymin"]
                staff_height = list_of_dictionaries[k]["staff_height"]
                plt.axhline(y=staff_ymin, color=color_scheme[k], linestyle='-', linewidth = 0.7)
                plt.axhline(y=staff_ymin +staff_height, color=color_scheme[k], linestyle='-', linewidth = 0.7)         
                boxes =  list_of_dictionaries[k]["boxes"]
                classes= list_of_dictionaries[k]["classes"]
                for i in range(0, len(boxes)):        
                    obj = boxes[i]
                    class_index= classes[i]
                    x,y,w,h = obj[1],obj[0],obj[3]-obj[1],obj[2]-obj[0]
                    color_label = color_dictionary[inv_label_dict[class_index]]
                    axes.add_patch(Rectangle((x, y), w, h, facecolor=color_label,linewidth = 0.7, edgecolor= color_label, fill=False))
        else:
            staff_ymin = list_of_dictionaries[k]["ymin"]
            staff_height = list_of_dictionaries[k]["staff_height"]
            plt.axhline(y=staff_ymin, color=color_scheme[k], linestyle='-', linewidth = 0.7)
            plt.axhline(y=staff_ymin +staff_height, color=color_scheme[k], linestyle='-', linewidth = 0.7)         
            boxes =  list_of_dictionaries[k]["boxes"]
            classes= list_of_dictionaries[k]["classes"]
            for i in range(0, len(boxes)):        
                obj = boxes[i]
                class_index= classes[i]
                x,y,w,h = obj[1],obj[0],obj[3]-obj[1],obj[2]-obj[0]
                color_label = color_dictionary[inv_label_dict[class_index]]
                axes.add_patch(Rectangle((x, y), w, h, facecolor=color_label,linewidth = 0.7, edgecolor= color_label, fill=False)) 
 
def visualize_parsed_staff(img_path,boxes,labels,margins,most_voted_middle,corrected_inferred_staffs_in_slices,\
                           scale_factor_width,scale_factor_height,symbol_index_around_middle_was_found,write):
    gray = False
    image= rescale_image(img_path,gray,scale_factor_width,scale_factor_height)
    fig = plt.figure(None, (10, 10), 200)
    fig.add_subplot(111)
    axes = fig.axes[0]
    axes.imshow(image,cmap="gray")
#     plt.axis('off')
    line_color = 'lime'
    text_color = 'red'
    plt.axhline(y=margins[0], color=line_color , linestyle='-', linewidth=0.75)
    plt.axhline(y=most_voted_middle, color='fuchsia' , linestyle='-', linewidth=0.75)
    plt.axhline(y=margins[1], color=line_color , linestyle='-', linewidth=0.75)
    labels = [inv_label_dict[label] for label in labels]
    lines = []
    print('symbol_index_around_middle_was_found yyy', symbol_index_around_middle_was_found)

    for i in range(0, len(labels)):        
        obj = boxes[i]
        class_index= labels[i]        
        x,y,w,h = obj[1],obj[0],obj[3]-obj[1],obj[2]-obj[0]
        print('i',i,'label', labels[i])
        label = labels[i]
        if '__' in labels[i]:
            label = label.split('__')[-1]
            print('label =', label)
        color_label = color_dictionary[label]
        axes.add_patch(Rectangle((x, y), w, h, edgecolor= color_label,linewidth = 1, fill=False, label=label))
        xmin = x
        xmax = x+w
        if i == symbol_index_around_middle_was_found:
            print('symbol_index_around_middle_was_found:', i)
            for j in [1,3,5,7,9,11,13]:
                lines.append([(xmin,corrected_inferred_staffs_in_slices[i][j]),\
                              (xmax,corrected_inferred_staffs_in_slices[i][j])])
        else:
            for j in [3,5,7,9,11]:
                lines.append([(xmin,corrected_inferred_staffs_in_slices[i][j]),\
                              (xmax,corrected_inferred_staffs_in_slices[i][j])])            
    c ='cyan' 
    lc = mc.LineCollection(lines, colors=c, linewidths=1)
    axes.add_collection(lc)                
        
    if write: 
        save_name = img_path.split('.jp')[0]+'.pdf'
 
 
def show_parsed_staff(img_path , staffn,scale_factor_width,scale_factor_height,write):


    img_name = img_path.split('/')[-1]
#     input_folder = img_path.split(img_name)[0]
    # choose a detection mode
    
    CONFIDENCE_THRESHOLD = 0.15
    OVERLAP_THRESHOLD = 0.4
    # implement the above 
    clean_dict=  Duration_Dictionaries(img_path, CONFIDENCE_THRESHOLD , OVERLAP_THRESHOLD)   
    list_of_dict= clean_dict.get_corrected_boxes_and_classes()
    list_of_dict=rescale_dictionaries(list_of_dict,scale_factor_width,scale_factor_height)
    boxes = list_of_dict[staffn]['boxes']
    labels = list_of_dict[staffn]['classes']
    margins = [list_of_dict[staffn]['ymin'], list_of_dict[staffn]['ymin']+ list_of_dict[staffn]['staff_height']]
    staff_dictionary = list_of_dict[staffn]
    inter_stave_height = staff_dictionary['inter_staff_line_spacing']
    print('interstaff spacing currenct',inter_stave_height)
    
    for i in range(0,len(list_of_dict)):
        inter_stave_height = list_of_dict[i]['inter_staff_line_spacing']
        print('i iss',inter_stave_height)
    grayscale= cv2.imread(img_path,0)
    nslices = 7
    middle_finder= Staff_Middle_Finder(grayscale,staff_dictionary,nslices)
    most_voted_middle, start_index = middle_finder.get_middle()
    list_of_stave_lines_per_stave_slice = middle_finder.get_detected_lines_per_staff_slice()
    symbol_indices = middle_finder.get_symbol_indices()
    print('most_voted middle',most_voted_middle)
    nlookahead = 9
    staff_reconstructer =Staff_Reconstructer(list_of_stave_lines_per_stave_slice,\
                                             inter_stave_height,most_voted_middle, start_index,symbol_indices,nlookahead)
    corrected_inferred_staffs_in_slices= staff_reconstructer.get_inferred_staff_lines()
    symbol_index_around_middle_was_found = staff_reconstructer.get_symbol_index_closest_to_start_index()
        #visualize
    write = False
    
    visualize_parsed_staff(img_path,boxes,labels,margins,most_voted_middle,corrected_inferred_staffs_in_slices,\
                           scale_factor_width,scale_factor_height,symbol_index_around_middle_was_found,write)
 
                
def show_all_parsed_staffs(img_path,scale_factor_width,scale_factor_height,write):
    img_name = img_path.split('/')[-1]    
    CONFIDENCE_THRESHOLD = 0.15
    OVERLAP_THRESHOLD = 0.4
    nslices = 7
    nlookahead = 9
    # implement the above 
    clean_dict=  Duration_Dictionaries(img_path, CONFIDENCE_THRESHOLD , OVERLAP_THRESHOLD)   
    list_of_dict= clean_dict.get_corrected_boxes_and_classes()
    list_of_dict=rescale_dictionaries(list_of_dict,scale_factor_width,scale_factor_height)
    grayscale= cv2.imread(img_path,0)
    
    list_of_boxes = []
    list_of_margins= []
    list_of_labels = []
    list_of_corrected_inferred_staffs_in_slices = [] # 
    list_of_symbol_index_around_middle_was_found = []
    list_of_middle_estimates =[]
    for staff_dictionary in list_of_dict:
        boxes =  staff_dictionary['boxes']
        list_of_boxes.append(boxes)
        labels =  staff_dictionary['classes']
        labels = [inv_label_dict[label] for label in labels]
        list_of_labels.append(labels)
        margins = [ staff_dictionary['ymin'], staff_dictionary['ymin']+  staff_dictionary['staff_height']]
        list_of_margins.append(margins)
        inter_stave_height =  staff_dictionary['inter_staff_line_spacing']    
        middle_finder= Staff_Middle_Finder(grayscale,staff_dictionary,nslices)
        most_voted_middle, start_index = middle_finder.get_middle()
        list_of_middle_estimates.append(most_voted_middle)
        list_of_stave_lines_per_stave_slice = middle_finder.get_detected_lines_per_staff_slice()
        symbol_indices = middle_finder.get_symbol_indices()
        staff_reconstructer =Staff_Reconstructer(list_of_stave_lines_per_stave_slice,\
                            inter_stave_height,most_voted_middle, start_index,symbol_indices,nlookahead)
        corrected_inferred_staffs_in_slices= staff_reconstructer.get_inferred_staff_lines()
        list_of_corrected_inferred_staffs_in_slices.append(corrected_inferred_staffs_in_slices)
        symbol_index_around_middle_was_found = staff_reconstructer.get_symbol_index_closest_to_start_index()
        list_of_symbol_index_around_middle_was_found.append(symbol_index_around_middle_was_found)
        
    gray = False
    image= rescale_image(img_path,gray,scale_factor_width,scale_factor_height)
    fig = plt.figure(None, (10, 10), 200)
    fig.add_subplot(111)
    axes = fig.axes[0]
    axes.imshow(image,cmap="gray")
    plt.axis('off')

    text_color = 'red'
    color_dict = {0: 'black',1: 'gold', 2: 'crimson', 3: 'cyan', 4: 'pink'}
    lines = []
    for j in range(len(list_of_dict)):
        boxes = list_of_boxes[j]
        labels = list_of_labels[j]
        margin=list_of_margins[j]
        middle_estimate = list_of_middle_estimates[j]
        corrected_inferred_staffs_in_slices = list_of_corrected_inferred_staffs_in_slices[j]
        plt.axhline(y=margin[0], color=color_dict[j%len(color_dict)], linestyle='-', linewidth=0.75)
        plt.axhline(y=middle_estimate, color='fuchsia' , linestyle='-', linewidth=0.75)
        plt.axhline(y=margin[1], color=color_dict[j%len(color_dict)], linestyle='-', linewidth=0.75)
        for i in range(0, len(labels)):        
            obj = boxes[i]
            label = labels[i]
            x,y,w,h = obj[1],obj[0],obj[3]-obj[1],obj[2]-obj[0]
#             print('labels i',labels[i])
            if '_' in labels[i]:
                label = label.split('_')[-1]
            color_label = color_dictionary[label]
            axes.add_patch(Rectangle((x, y), w, h, edgecolor= color_label,linewidth = 1, fill=False)) 
            xmin = x
            xmax = x+w
            if i == symbol_index_around_middle_was_found:
#                 print('symbol_index_around_middle_was_found:', i)
                for j in [1,3,5,7,9,11,13]:
                    lines.append([(xmin,corrected_inferred_staffs_in_slices[i][j]),\
                                  (xmax,corrected_inferred_staffs_in_slices[i][j])])
            else:
                for j in [3,5,7,9,11]:
                    lines.append([(xmin,corrected_inferred_staffs_in_slices[i][j]),\
                                  (xmax,corrected_inferred_staffs_in_slices[i][j])])            
        c ='cyan' 
        lc = mc.LineCollection(lines, colors=c, linewidths=1)
        axes.add_collection(lc)  
    if write: 
        save_name = img_path.split('.jp')[0]+'.pdf'
        plt.savefig(save_name)                           
            
def show_all_staffs(input_folder, number,scale_factor_width,scale_factor_height,write):

    image_paths=glob.glob(input_folder+'/'"*.jpg")
    image_paths+=glob.glob(input_folder+'/'"*.jpeg") 
    img_path =image_paths[number] 
    print('image path=====:', img_path)

    img_name = img_path.split('/')[-1]
    input_folder = img_path.split(img_name)[0]
    # choose a detection mode
    
    CONFIDENCE_THRESHOLD = 0.1
    OVERLAP_THRESHOLD = 0.4
    # implement the above 
    clean_dict=  Duration_Dictionaries(input_folder,img_name, CONFIDENCE_THRESHOLD , OVERLAP_THRESHOLD)   
    list_of_dict= clean_dict.get_corrected_boxes_and_classes()
    list_of_dict=rescale_dictionaries(list_of_dict,scale_factor_width,scale_factor_height)
    list_of_boxes= []
    list_of_pos = []
    list_of_labels = []
    list_of_margins = []
    list_of_middle_estimates= []
    for staffn in range(len(list_of_dict)):
        pitch_retriever=Positions_on_staff(img_path,list_of_dict,scale_factor_width,scale_factor_height)
        pitch_retriever.parse_detected_symbols(staffn)
        middle_estimate= pitch_retriever.get_middle_estimate()
        list_of_middle_estimates.append(middle_estimate)
        margins=[list_of_dict[staffn]['ymin'], list_of_dict[staffn]['ymin'] +  list_of_dict[staffn]['staff_height'] ]
        list_of_margins.append(margins)
        labels=pitch_retriever.get_labels()
#         classes_and_probs=pitch_retriever.get_classes_and_probs()
        boxes=pitch_retriever.get_all_boxes()
        list_of_boxes.append(boxes)
        pos= pitch_retriever.get_positions_on_staff()
        list_of_pos.append(pos)
        list_of_labels.append(labels)
#         foregrounds = pitch_retriever.get_foregrounds()
#         dur_probs = pitch_retriever.get_dur_probabilities_for_current_staff()        
    
    gray = False
    image= rescale_image(img_path,gray,scale_factor_width,scale_factor_height)
    fig = plt.figure(None, (10, 10), 200)
    fig.add_subplot(111)
    axes = fig.axes[0]
    axes.imshow(image,cmap="gray")
    plt.axis('off')

    text_color = 'red'
    color_dict = {0: 'black',1: 'gold', 2: 'crimson', 3: 'cyan', 4: 'pink'}
#     plt.axhline(y=393, color=line_color , linestyle='-', linewidth=0.75) 
    for j in range(len(list_of_boxes)):
        boxes = list_of_boxes[j]
        pos=list_of_pos[j]
        labels = list_of_labels[j]
        margin=list_of_margins[j]
        middle_estimate = list_of_middle_estimates[j]
        plt.axhline(y=margin[0], color=color_dict[j%len(color_dict)], linestyle='-', linewidth=0.75)
        plt.axhline(y=middle_estimate, color='fuchsia' , linestyle='-', linewidth=0.75)
        plt.axhline(y=margin[1], color=color_dict[j%len(color_dict)], linestyle='-', linewidth=0.75)
        for i in range(0, len(labels)):        
            obj = boxes[i]
            class_index= labels[i]

            x,y,w,h = obj[1],obj[0],obj[3]-obj[1],obj[2]-obj[0]
    #         print('i',i,'label', labels[i], 'dur_prob', dur_probs[i],'posNprob',classes_and_probs[i] ,'fg',foregrounds[i])
            if 'void' not in labels[i]:
                label = labels[i]
                print('label',label)
                if '__' in labels[i]:
                    label = label.split('__')[-1]
    #                 print('label =', label)
                color_label = color_dictionary[label]
                axes.add_patch(Rectangle((x, y), w, h, edgecolor= color_label,linewidth = 1, fill=False, label=pos[i])) 
                plt.text(x,y,str(pos[i]),fontsize=7,color =text_color)  

    if write: 
        save_name = img_path.split('.jp')[0]+'.pdf'
        plt.savefig(save_name)                       	
	
