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
    #  XX 'no_ResNet1d_FF'      XX         #
    ########################################

# Calculation

ABLATION = 'no_ResNet2d'
families = 'telomerase'
EXP = f'{ABLATION}-{families}'

TEST_FILE = '../data/test_telomerase.csv'

tests = ['test-1/','test-2/','test-3/']
for TEST in tests:
    for i in range(2):
        if i ==0: WEIGHTS_PATH = TEST + EXP  
        if i ==1: WEIGHTS_PATH = TEST + EXP + '/valid_with_test'
            # Args     
        
        ABLATION_WEIGHTS = WEIGHTS_PATH + '/weights.pmt'
        OUT_FILE = WEIGHTS_PATH +'/test_log.csv'

            
            # ABLATION: {ABLATION} \n
            # TEST_FILE: {TEST_FILE} \n
        print(f'''
            ABLATION_WEIGHTS: {ABLATION_WEIGHTS} \n
            OUT_PATH: {OUT_FILE} \n
            ''')
        
        ######################################
        # # # # Execution
        test_net(
            ablation= ABLATION,
            test_file= TEST_FILE,
            model_weights=ABLATION_WEIGHTS,
            output_file= OUT_FILE)
