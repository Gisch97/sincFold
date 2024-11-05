import os 
from sincfold.ablation.ablation_train import train_net

    ########################################
    # family:                              #
    #     'tRNA'                           #
    #     'telomerase'                     #
    #     'RNaseP'                         #
    #                                      #
    # ablations:                           #
    #     'sincfold'                       #
    #     'C1D_C2D'                        #
    #     'no_ResNet2d'                    #
    #     '1ResNet2d'                      #
    #     'no_ResNet1d'                    #
    #     'no_ResNet1d_FF'                 #
    ########################################

# Calculation
 
ablations = ['1ResNet2d', 'no_ResNet1d']
for ABLATION in ablations:
    print(ABLATION)
 
# Args    
TRAIN_FILE = '../../data/ArchiveII.csv'
OUT_PATH = './' + ABLATION
VALID_FILE=None 
    
print(f'''
    ABLATION: {ABLATION}, \n
    TRAIN_FILE: {TRAIN_FILE}, \n
    OUT_PATH: {OUT_PATH}, \n
    VALID_FILE: {VALID_FILE}, \n
    ''')
########################################
# # Execution
train_net(
ablation= ABLATION,
    train_file= TRAIN_FILE,
    out_path= OUT_PATH,
    valid_file= VALID_FILE)
