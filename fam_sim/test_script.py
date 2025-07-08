import os

fams = ["5s", "tmRNA", "tRNA", "srp", "grp1", "RNaseP", "23s", "telomerase", "16s"]
kinds = ["rnadist", "herarchical_cluster", "samples"]

STEPS = 2000
DATA_PATH_ref = "data/fam-fold/"
SAVE_PATH_ref = f"models/fam-sim/{STEPS}_steps/"

for fam in fams:
    test_file = f"{DATA_PATH_ref}test_{fam}.csv"
    out_path = f"{SAVE_PATH_ref}{fam}/"
    weigths_path = f"{SAVE_PATH_ref}{fam}/weights.pmt"
    test = f"sincFold -d cuda test {test_file} -w {weigths_path} -o {out_path}test_{STEPS}steps.csv"
    print(f"Running: {test}")
    os.system(test)


for kind in kinds:
    MODEL_PATH = f"models/{kind}/"

    for folder in ["dist_100/", "dist_200/", "dist_400/"]:
        MODEL = MODEL_PATH + folder + f"{STEPS}_steps/"

        # # original training
        for fam in fams:
            test_file = f"{DATA_PATH_ref}test_{fam}.csv"
            out_path = f"{MODEL}{fam}/"
            weigths_path = f"{MODEL}{fam}/weights.pmt"
            test = f"sincFold -d cuda test {test_file} -w {weigths_path} -o {out_path}test_{STEPS}steps.csv"
            print(f"Running: {test}")
            os.system(test)
