import os

fams = ["5s", "tmRNA", "tRNA", "srp", "grp1", "RNaseP", "23s", "telomerase", "16s"]
DATA_PATH = "data/original/"
MODELS_PATH = "models/original/"
# f'sincFold -d cuda train {train_file} --valid-file working_path/valid.csv -o working_path/output/'

for fam in fams:
    train_file = f"{DATA_PATH}train_{fam}.csv"
    out_path = f"{MODELS_PATH}{fam}/"
    train = f"sincFold -d cuda train {train_file} -o {out_path}"
    print(f"Running: {train}")
    os.system(train)
