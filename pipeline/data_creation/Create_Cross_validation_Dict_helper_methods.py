import os 
import numpy as np
import json
import glob 

def get_annotated_books(annotated_data_dir):
    IDEM_books = os.listdir(os.path.join(annotated_data_dir,'IDEM'))
    Non_IDEM_books = os.listdir(os.path.join(annotated_data_dir,'Non_IDEM'))   
    books = IDEM_books + Non_IDEM_books
    return books

def get_k_fold_train_test_val_partitions(books, train_val_split,k):
    np.random.shuffle(books)
    books_per_group = int((len(books)/k))
    groups = []
    i=0
    while i in range(len(books)):
        groups.append(books[i:i+books_per_group])
        i+=books_per_group
    train_test_val_partition_dicts={}
    for i in range(k):
        partition_dict={}
        test = groups[i]
        train_val = []
        for  j in range(len(groups)):
            if j != i:
                train_val.append(groups[j])
        train_val = [item for sublist in train_val for item in sublist]
        number_of_training_books = int(np.ceil(train_val_split*len(train_val)))
        train = train_val[:number_of_training_books]
        val = train_val[number_of_training_books:]
        partition_dict['train']= train
        partition_dict['val']= val
        partition_dict['test']= test
        train_test_val_partition_dicts[i] =partition_dict
    return train_test_val_partition_dicts

def get_train_val_test_images_dictionary(source_folder,selected_partition):
    idem_dir= os.path.join(source_folder, 'IDEM')
    non_idem_dir= os.path.join(source_folder, 'Non_IDEM')
    image_dict={}
    for mode in ['train','val','test']:
        all_mode_images = []
        for book in selected_partition[mode]:
            if book in os.listdir(idem_dir):
                image_source='IDEM'
            else:
                image_source ='Non_IDEM'
            book_images = glob.glob(os.path.join(source_folder,image_source,book)+'/**.jp*g')       
            all_mode_images.append(book_images)
        all_mode_images = [item for sublist in all_mode_images for item in sublist]
        image_dict[mode]= all_mode_images
    return image_dict

def write_cross_validation_dict(annotated_data_dir,train_val_split,k,dict_name):  
    books = get_annotated_books(annotated_data_dir)
    partitions_of_test_train_val_books = get_k_fold_train_test_val_partitions(books, train_val_split,k) 
    
    partitions_of_test_train_val_books['annotated_data_dir']= annotated_data_dir
    partitions_of_test_train_val_books['k'] = int(k)
    partitions_of_test_train_val_books['train_val_split'] = train_val_split    
    write_path= os.path.join(os.getcwd(),dict_name)
    print('wp',write_path)
    with open(write_path, 'w') as fp:
        json.dump(partitions_of_test_train_val_books, fp)
