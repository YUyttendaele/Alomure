import numpy as np
import cv2

def get_stave_line_candidates(img,box,inter_staff_line_spacing):        
        """
        We aim to finde the group of 5- stave lines within the given box which 
        is assumed to contain the staff
        """
        STAVE_LINE_IDENTFICATION_MARGIN =inter_staff_line_spacing*0.35    
        LINE_TO_BOX_WIDTH_IDENTIFICATION_FRACTION = 0.8
        adaptive_threshold_upper_range = 15
        adaptive_threshold_lower_range = 3    

        img_excerpt= img[box[0]:box[2], box[1]:box[3]]   
        
        excerpt_height, excerpt_width = img_excerpt.shape[0],img_excerpt.shape[1]
        list_of_candidate_stave_lines = []
        if excerpt_width>0 and excerpt_height>0:                  
            adaptive_thresholding_constant = adaptive_threshold_upper_range
            errors = [] 
            while adaptive_thresholding_constant >= adaptive_threshold_lower_range:
                for kernel_width in range(1,2):
                    binary =staves_binary(img_excerpt,int(inter_staff_line_spacing),
                                          adaptive_thresholding_constant,kernel_width)                     
                    # get histogram and find horizontal lines based on the given inter staff spacing (estimate)
                    histo =  get_histogram(binary) 
                    minimal_line_segment_length= LINE_TO_BOX_WIDTH_IDENTIFICATION_FRACTION *excerpt_width
                    stave_line_candidates = []
                    for line in range(0, histo.shape[0]-1):
                        if len(stave_line_candidates) ==0 and histo[line]>  minimal_line_segment_length:
                            stave_line_candidates.append(line)
                        elif len(stave_line_candidates) !=0 and histo[line]> minimal_line_segment_length\
                            and line-stave_line_candidates[-1] >STAVE_LINE_IDENTFICATION_MARGIN:
                            stave_line_candidates.append(line)                                                             
                    list_of_candidate_stave_lines.append(stave_line_candidates) 
                    error = staff_error(stave_line_candidates)
                    errors.append(error)
                if error > 0:
                    adaptive_thresholding_constant -=1
                else:
                    break
            minimal_error_index = np.argmin(errors)          
            return list_of_candidate_stave_lines[minimal_error_index]
        else:
            return list_of_candidate_stave_lines
    
def staves_binary(grayscale,inter_staff_line_spacing,adaptive_thresholding_constant,kernel_width):
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (inter_staff_line_spacing,kernel_width))
    horizontal = cv2.dilate(grayscale, horizontalStructure)
    if inter_staff_line_spacing %2 ==0:
        inter_staff_line_spacing +=1  
    binary_image_with_horizontal_lines =  cv2.adaptiveThreshold(horizontal,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,inter_staff_line_spacing,adaptive_thresholding_constant)   
    return binary_image_with_horizontal_lines 

def get_histogram(img):
    histo  =  []
    for i in range(0,img.shape[0]):
        counter = 0
        for k in range(0,img.shape[1]):
            if img[i][k]==0 :
                counter= counter+1
        histo.append(counter)       
    return np.asarray(histo)       
         
def staff_error(stave_lines):
        return np.abs(len(stave_lines)-5)  
