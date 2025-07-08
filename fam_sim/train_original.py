import os

fams = ["5s", "tmRNA", "tRNA", "srp", "grp1", "RNaseP", "23s", "telomerase", "16s"]
kinds = ["rnadist", "herarchical_cluster", "samples"]

steps_prev = 1000
steps_target = 2000
steps = steps_target - steps_prev


DATA_PATH_ref = "data/fam-fold/"
MODEL_PATH_ref = f"models/fam-sim/{steps_prev}_steps/"
SAVE_PATH_ref = f"models/fam-sim/{steps_target}_steps/"
os.makedirs(SAVE_PATH_ref, exist_ok=True)


# SEARCH FOR TARGET DATASETS (REF), TRAIN
for fam in fams:
    train_file = f"{DATA_PATH_ref}train_{fam}.csv"
    valid_file = f"{DATA_PATH_ref}valid_{fam}.csv"
    out_path = f"{SAVE_PATH_ref}{fam}/"
    ################# TRAIN FROM ZERO #####################
    # train = f"sincFold -d cuda train {train_file} --valid-file {valid_file} -o {out_path} -n {steps}"

    ################# TRAIN WITH WEIGTHS #####################
    weigths_path = f"{MODEL_PATH_ref}{fam}/weights.pmt"
    train = f"sincFold -d cuda train {train_file} --valid-file {valid_file} -o {out_path} -w {weigths_path} -n {steps}"
    ##### EXEC #########
    print(f"Running: {train}")
    os.system(train)

# SET PATHS TO MODELS / SAVE
for kind in kinds:

    DATA_PATH = f"data/{kind}/"
    SAVE_PATH = f"models/{kind}/"
    for folder in ["dist_100/", "dist_200/", "dist_400/"]:
        DATA = DATA_PATH + folder
        MODEL = SAVE_PATH + folder + f"{steps_prev}_steps/"
        SAVE = SAVE_PATH + folder + f"{steps_target}_steps/"
        os.makedirs(SAVE_PATH, exist_ok=True)
        os.makedirs(SAVE_PATH + folder, exist_ok=True)
        os.makedirs(SAVE, exist_ok=True)
        # SEARCH FOR TARGET DATASETS, TRAIN
        for fam in fams:
            train_file = f"{DATA}train_{fam}.csv"
            valid_file = f"{DATA}valid_{fam}.csv"
            out_path = f"{SAVE}{fam}/"

            ################# TRAIN FROM ZERO #####################
            # train = f"sincFold -d cuda train {train_file} --valid-file {valid_file} -o {out_path} -n {steps}"

            ################# TRAIN WITH WEIGTHS #####################
            weigths_path = f"{MODEL}{fam}/weights.pmt"
            train = f"sincFold -d cuda train {train_file} --valid-file {valid_file} -o {out_path} -w {weigths_path} -n {steps}"
            ##### EXEC #########
            print(f"Running: {train}")
            os.system(train)
