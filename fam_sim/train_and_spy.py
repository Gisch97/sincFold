#!/usr/bin/env python

import os
import sys

###############################################################################
# CONFIG
###############################################################################
fams = ["5s", "tmRNA", "tRNA", "srp", "grp1", "RNaseP", "23s", "telomerase", "16s"]
kinds = ["rnadist", "herarchical_cluster", "samples"]

START_EPOCH = 1  # último checkpoint disponible
END_EPOCH = 15  # epoch final (inclusive).

DATA_PATH_FAM = "data/fam-fold/"
MODEL_ROOT_FAM = "models/train_and_spy/fam-sim/"  # coincide con tu script de test
DATA_PATH_KIND = {k: f"data/{k}/" for k in kinds}
MODEL_ROOT_KIND = {k: f"models/train_and_spy/{k}/" for k in kinds}

DIST_FOLDERS = ["dist_100/", "dist_200/", "dist_400/"]  # subcarpetas para kinds


###############################################################################
# UTILIDADES
###############################################################################
def run_cmd(cmd: str):
    """Imprime y ejecuta un comando shell."""
    print(f"\n$ {cmd}\n")
    sys.stdout.flush()
    os.system(cmd)


def train_one_epoch(train_csv, valid_csv, out_dir, prev_weights=None):
    """Genera la llamada sincFold para entrenar 1 época."""
    if prev_weights and os.path.exists(prev_weights):
        cmd = (
            f"sincFold -d cuda train {train_csv} "
            f"--valid-file {valid_csv} -o {out_dir} "
            f"-w {prev_weights} -n 1"
        )
    else:
        cmd = (
            f"sincFold -d cuda train {train_csv} "
            f"--valid-file {valid_csv} -o {out_dir} -n 1"
        )
    run_cmd(cmd)


def test_epoch(test_csv, weights, out_path):
    """Genera la llamada sincFold para testear un modelo dado."""
    if not os.path.exists(weights):
        print(f"[WARN] Pesos no encontrados: {weights}")
        return
    cmd = f"sincFold -d cuda test {test_csv} " f"-w {weights} -o {out_path}test.csv"
    run_cmd(cmd)


###############################################################################
# BUCLE PRINCIPAL (incluye la época 0)
###############################################################################
for target_epoch in range(START_EPOCH, END_EPOCH + 1):
    prev_epoch = target_epoch - 1  # puede ser 0 cuando target_epoch == 0
    epoch_tag = f"{target_epoch}epoch"

    print("\n" + "=" * 70)
    print(f"      >>>>>  Época {target_epoch}  <<<<<")
    print("=" * 70)

    # ----------- 1) fam‑sim ---------------------------------------------------
    save_root_fam = f"{MODEL_ROOT_FAM}{target_epoch}_epoch/"
    prev_root_fam = f"{MODEL_ROOT_FAM}{prev_epoch}_epoch/" if prev_epoch > 0 else None
    os.makedirs(save_root_fam, exist_ok=True)
    for fam in fams:
        train_csv = f"{DATA_PATH_FAM}train_{fam}.csv"
        valid_csv = f"{DATA_PATH_FAM}valid_{fam}.csv"
        fam_out = f"{save_root_fam}{fam}/"

        prev_w = (
            f"{prev_root_fam}{fam}/weights.pmt"  # solo si prev_root_fam existe
            if prev_root_fam
            else None
        )
        # ---- train ----
        train_one_epoch(train_csv, valid_csv, fam_out, prev_w)

        #         # ---- test ----
        test_csv = f"{DATA_PATH_FAM}test_{fam}.csv"
        weights = f"{fam_out}weights.pmt"
        test_epoch(test_csv, weights, fam_out)

    # # # ----------- 2) kinds ------------------------------------------------------
    for kind in kinds:
        data_path = DATA_PATH_KIND[kind]
        model_root = MODEL_ROOT_KIND[kind]

        for folder in DIST_FOLDERS:
            save_root_kind = f"{model_root}{folder}{target_epoch}_epoch/"
            prev_root_kind = (
                f"{model_root}{folder}{prev_epoch}_epoch/" if prev_epoch > 0 else None
            )
            os.makedirs(save_root_kind, exist_ok=True)
            for fam in fams:
                train_csv = f"{data_path}{folder}train_{fam}.csv"
                valid_csv = f"{data_path}{folder}valid_{fam}.csv"
                fam_out = f"{save_root_kind}{fam}/"

                prev_w = (
                    f"{prev_root_kind}{fam}/weights.pmt" if prev_root_kind else None
                )

                train_one_epoch(train_csv, valid_csv, fam_out, prev_w)

                test_csv = f"{DATA_PATH_FAM}test_{fam}.csv"
                weights = f"{fam_out}weights.pmt"
                test_epoch(test_csv, weights, fam_out)
