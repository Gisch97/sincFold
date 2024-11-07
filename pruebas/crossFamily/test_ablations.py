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
ablations =  ['C1D_C2D', 'no_ResNet2d', 'sincfold']
for ABLATION in ablations:
    for family in families:
        # Args    
        TEST_FILE = TEST_FAMILY[family]
        ABLATION_WEIGHTS = os.path.join(os.path.join(CROSS_FAMILIES_WEIGHTS,family), ABLATION) + '/valid_with_test/weights.pmt'
        OUT_FILE = CROSS_FAMILIES_WEIGHTS + family + '/' + ABLATION +'/test_log.csv'

            
        print(f'''
            ABLATION: {ABLATION} \n
            TEST_FILE: {TEST_FILE} \n
            ABLATION_WEIGHTS: {ABLATION_WEIGHTS} \n
            OUT_PATH: {OUT_FILE} \n
            ''')
        
        ######################################
        # # # Execution
        test_net(
            ablation= ABLATION,
            test_file= TEST_FILE,
            model_weights=ABLATION_WEIGHTS,
            output_file= OUT_FILE)