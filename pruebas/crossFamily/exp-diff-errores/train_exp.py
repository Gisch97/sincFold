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

ABLATION = 'no_ResNet2d'                    
valid_with_test=False

########################################
# Paths definitions
TEST = 'test-1/'
families = 'telomerase'
TRAIN_LOG = 'train_log.csv'
EXP = f'{ABLATION}-{families}'
   


TRAIN_FAMILY = '../data/train_telomerase.csv'
TEST_FAMILY = '../data/test_telomerase.csv'
# F = CROSS_FAMILIES_WEIGHTS + family +'/'+ABLATION
########################################
# for family in families:
# Args    
TRAIN_FILE = TRAIN_FAMILY

for i in range(2):
    # if i ==1: valid_with_test=True
    if i ==1: TEST = 'test-2/'
    VALID_FILE=None
    OUT_PATH = TEST + EXP 
    if valid_with_test:
        OUT_PATH+= '/valid_with_test/'
        VALID_FILE= TEST_FAMILY
        
    print(f'''
        ABLATION: {ABLATION}, \n
        TRAIN_FILE: {TRAIN_FILE}, \n
        OUT_PATH: {OUT_PATH}, \n
        VALID_FILE: {VALID_FILE}, \n
        NUM EPOCH: 5, \n
        ''')
        ########################################
    #     # # Execution
    train_net(
    ablation= ABLATION,
        train_file= TRAIN_FILE,
        out_path= OUT_PATH,
        valid_file= VALID_FILE,
        num_epoch=5)
