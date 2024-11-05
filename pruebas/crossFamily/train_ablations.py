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

# family = 'RNaseP'

ABLATIONS = 'sincfold'
valid_with_test=True

for ABLATION in ABLATIONS:
    ########################################
    # Paths definitions

    CROSS_FAMILIES_PATH = 'data/'   
    CROSS_FAMILIES_WEIGHTS = 'weights/'
    TRAIN_LOG = 'train_log.csv'

    families = ['tRNA', 'telomerase', 'RNaseP']
    TRAIN_FAMILY = {fam: CROSS_FAMILIES_PATH + 'train_' + fam + '.csv' for fam in families}
    TEST_FAMILY = {fam: CROSS_FAMILIES_PATH + 'test_' + fam + '.csv' for fam in families}

    # F = CROSS_FAMILIES_WEIGHTS + family +'/'+ABLATION
    ########################################
    for family in families:
        # Args    
        TRAIN_FILE = TRAIN_FAMILY[family]
        OUT_PATH = CROSS_FAMILIES_WEIGHTS + family + '/' + ABLATION
        VALID_FILE=None
        if valid_with_test:
            OUT_PATH+= '/valid_with_test/'
            VALID_FILE= TEST_FAMILY[family]
            
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
