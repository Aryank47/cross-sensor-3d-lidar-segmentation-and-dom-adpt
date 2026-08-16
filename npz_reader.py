from pathlib import Path

import numpy as np


def dump_numpy_files(file_paths, output_file="numpy_contents.txt"):
    """
    Read multiple .npy or .npz files and write their contents to a text file.

    Supports:
        .npy -> single NumPy array
        .npz -> archive containing one or more NumPy arrays
    """

    with open(output_file, "w", encoding="utf-8") as out:

        for file_path in file_paths:
            file_path = Path(file_path)

            out.write("=" * 100 + "\n")
            out.write(f"FILE: {file_path}\n")
            out.write("=" * 100 + "\n")

            if not file_path.exists():
                out.write("ERROR: File does not exist.\n\n")
                continue

            try:
                # ------------------------------------------------------------------
                # .NPY FILE
                # ------------------------------------------------------------------
                if file_path.suffix.lower() == ".npy":
                    arr = np.load(file_path, allow_pickle=True)

                    out.write("TYPE: NPY\n")
                    out.write(f"PYTHON TYPE: {type(arr)}\n")
                    out.write(f"SHAPE: {arr.shape}\n")
                    out.write(f"DTYPE: {arr.dtype}\n")
                    out.write("CONTENTS:\n")

                    out.write(np.array2string(arr, threshold=np.inf, max_line_width=200))

                    out.write("\n")

                # ------------------------------------------------------------------
                # .NPZ FILE
                # ------------------------------------------------------------------
                elif file_path.suffix.lower() == ".npz":

                    with np.load(file_path, allow_pickle=True) as data:

                        out.write("TYPE: NPZ\n")
                        out.write(f"KEYS: {data.files}\n\n")

                        for key in data.files:
                            arr = data[key]

                            out.write("-" * 80 + "\n")
                            out.write(f"KEY: {key}\n")
                            out.write(f"PYTHON TYPE: {type(arr)}\n")
                            out.write(f"SHAPE: {arr.shape}\n")
                            out.write(f"DTYPE: {arr.dtype}\n")
                            out.write("CONTENTS:\n")

                            out.write(np.array2string(arr, threshold=np.inf, max_line_width=200))

                            out.write("\n\n")

                else:
                    out.write(f"ERROR: Unsupported file extension " f"'{file_path.suffix}'\n")

            except Exception as exc:
                out.write(f"ERROR while reading file: " f"{type(exc).__name__}: {exc}\n")

            out.write("\n\n")

    print(f"Output written to: {output_file}")


if __name__ == "__main__":

    file_paths = [
        # "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_dales_test_37185/confusion_common.npy",
        # "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_dales_test_38515/confusion_common.npy",
        # "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_eclair_test_37189/confusion_common.npy",
        # "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_eclair_test_37189/confusion_train_space.npy",
        "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_eclair_pub_m1_s1_e2e_w120_test_39098/confusion_common.npy",
        "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_eclair_pub_m1_s1_e2e_w120_test_39098/confusion_train_space.npy",
        "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_dales_pub_m1_s1_e2d_w120_test_39099/confusion_common.npy",
        "/scratch/m23csa510/cross_eval_runs/clean_eval/dales_to_dales_pub_m1_s1_d2d_w120_test_39100/confusion_common.npy",
        "/scratch/m23csa510/cross_eval_runs/clean_eval/dales_to_dales_pub_m1_s1_d2d_w120_test_39100/confusion_train_space.npy",
        "/scratch/m23csa510/cross_eval_runs/clean_eval/dales_to_eclair_pub_m1_s1_d2e_w120_test_39101/confusion_common.npy",
    ]

    dump_numpy_files(file_paths=file_paths, output_file="confusion_matrices.txt")
