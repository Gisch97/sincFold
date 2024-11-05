import os 
from sincfold.ablation.ablation_test import test_net

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

family = 'RNaseP'

########################################
# Paths definitions

CROSS_FAMILIES_PATH = 'data/'   
CROSS_FAMILIES_WEIGHTS = 'weights/'
TRAIN_LOG = 'train_log.csv'
families = ['tRNA', 'telomerase', 'RNaseP']

TEST_FAMILY = {fam: CROSS_FAMILIES_PATH + 'test_' + fam + '.csv' for fam in families}

########################################
ABLATION = 'no_ResNet2d'
for family in families:
    # Args    
    TEST_FILE = TEST_FAMILY[family]
    ABLATION_WEIGHTS = os.path.join(os.path.join(CROSS_FAMILIES_WEIGHTS,family), ABLATION) + '/weights.pmt'
    OUT_PATH = CROSS_FAMILIES_WEIGHTS + family + '/' + ABLATION

        
    print(f'''
        ABLATION: {ABLATION} \n
        TEST_FILE: {TEST_FILE} \n
        OUT_PATH: {OUT_PATH} \n
        ''')
    ########################################
    # # # Execution
    # test_net(
    # ablation= ABLATION,
    #     test_file= TEST_FILE,
    #     out_path= OUT_PATH)