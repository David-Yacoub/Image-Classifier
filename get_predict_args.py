# PROGRAMMER: David Yacoub
# DATE CREATED: 28 Aug 2021                                   
# REVISED DATE: 

##
import argparse

# 
def get_predict_args():
    # Create Parse using ArgumentParser
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_image', type = str, default = 'flowers/test/1/image_06743.jpg', 
                    help = 'path_to_image')
    parser.add_argument('--top_k', type = int, default = 3, 
                    help = 'Return top KK most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'a mapping of categories to real names')
    parser.add_argument('--chkpoint', type = str, default = 'save_dir/checkpoint.pth', 
                    help = 'checkpoint path')
    parser.add_argument('--device', type = str, default = 'cuda', 
                    help = 'Use GPU for training')
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()
