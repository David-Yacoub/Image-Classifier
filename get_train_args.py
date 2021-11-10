# PROGRAMMER: David Yacoub
# DATE CREATED: 28 Aug 2021                                   
# REVISED DATE: 

##
import argparse

# 
def get_train_args():
    # Create Parse using ArgumentParser
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type = str, default = 'save_dir/checkpoint.pth', 
                    help = 'Set directory to save checkpoints')
    parser.add_argument('--arch', type = str, default = 'densenet', 
                    help = 'Choose architecture')
    parser.add_argument('--learning_rate', type = int, default = 0.0001, 
                    help = 'hyperparameters: learning_rate')
    parser.add_argument('--hidden_units', type = list, default = [512], 
                    help = 'hyperparameters: hidden_units')
    parser.add_argument('--epochs', type = int, default = 13, 
                    help = 'hyperparameters: epochs')
    parser.add_argument('--device', type = str, default = 'cuda', 
                    help = 'Use GPU for training')
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()
