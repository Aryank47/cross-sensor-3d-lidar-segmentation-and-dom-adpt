# from pathlib import Path

# import numpy as np


# def dump_numpy_files(file_paths, output_file="numpy_contents.txt"):
#     """
#     Read multiple .npy or .npz files and write their contents to a text file.

#     Supports:
#         .npy -> single NumPy array
#         .npz -> archive containing one or more NumPy arrays
#     """

#     with open(output_file, "w", encoding="utf-8") as out:

#         for file_path in file_paths:
#             file_path = Path(file_path)

#             out.write("=" * 100 + "\n")
#             out.write(f"FILE: {file_path}\n")
#             out.write("=" * 100 + "\n")

#             if not file_path.exists():
#                 out.write("ERROR: File does not exist.\n\n")
#                 continue

#             try:
#                 # ------------------------------------------------------------------
#                 # .NPY FILE
#                 # ------------------------------------------------------------------
#                 if file_path.suffix.lower() == ".npy":
#                     arr = np.load(file_path, allow_pickle=True)

#                     out.write("TYPE: NPY\n")
#                     out.write(f"PYTHON TYPE: {type(arr)}\n")
#                     out.write(f"SHAPE: {arr.shape}\n")
#                     out.write(f"DTYPE: {arr.dtype}\n")
#                     out.write("CONTENTS:\n")

#                     out.write(np.array2string(arr, threshold=np.inf, max_line_width=200))

#                     out.write("\n")

#                 # ------------------------------------------------------------------
#                 # .NPZ FILE
#                 # ------------------------------------------------------------------
#                 elif file_path.suffix.lower() == ".npz":

#                     with np.load(file_path, allow_pickle=True) as data:

#                         out.write("TYPE: NPZ\n")
#                         out.write(f"KEYS: {data.files}\n\n")

#                         for key in data.files:
#                             arr = data[key]

#                             out.write("-" * 80 + "\n")
#                             out.write(f"KEY: {key}\n")
#                             out.write(f"PYTHON TYPE: {type(arr)}\n")
#                             out.write(f"SHAPE: {arr.shape}\n")
#                             out.write(f"DTYPE: {arr.dtype}\n")
#                             out.write("CONTENTS:\n")

#                             out.write(np.array2string(arr, threshold=np.inf, max_line_width=200))

#                             out.write("\n\n")

#                 else:
#                     out.write(f"ERROR: Unsupported file extension " f"'{file_path.suffix}'\n")

#             except Exception as exc:
#                 out.write(f"ERROR while reading file: " f"{type(exc).__name__}: {exc}\n")

#             out.write("\n\n")

#     print(f"Output written to: {output_file}")


# if __name__ == "__main__":

#     file_paths = [
#         # "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_dales_test_37185/confusion_common.npy",
#         # "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_dales_test_38515/confusion_common.npy",
#         # "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_eclair_test_37189/confusion_common.npy",
#         # "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_eclair_test_37189/confusion_train_space.npy",
#         "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_eclair_pub_m1_s1_e2e_w120_test_39098/confusion_common.npy",
#         "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_eclair_pub_m1_s1_e2e_w120_test_39098/confusion_train_space.npy",
#         "/scratch/m23csa510/cross_eval_runs/clean_eval/eclair_to_dales_pub_m1_s1_e2d_w120_test_39099/confusion_common.npy",
#         "/scratch/m23csa510/cross_eval_runs/clean_eval/dales_to_dales_pub_m1_s1_d2d_w120_test_39100/confusion_common.npy",
#         "/scratch/m23csa510/cross_eval_runs/clean_eval/dales_to_dales_pub_m1_s1_d2d_w120_test_39100/confusion_train_space.npy",
#         "/scratch/m23csa510/cross_eval_runs/clean_eval/dales_to_eclair_pub_m1_s1_d2e_w120_test_39101/confusion_common.npy",
#     ]

#     dump_numpy_files(file_paths=file_paths, output_file="confusion_matrices.txt")


# from pathlib import Path
# import numpy as np


# BASE_FOLDER = Path(
#     "/scratch/m23csa510/cross_eval_runs/clean_eval"
# )

# OUTPUT_FILE = "confusion_matrices.txt"


# # Recursively find all confusion .npy files
# files = sorted(BASE_FOLDER.rglob("confusion_*.npy"))

# print(f"Found {len(files)} confusion matrix files.")


# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

#     for path in files:
#         try:
#             arr = np.load(path)

#             # Run/folder name
#             run_name = path.parent.name

#             f.write("=" * 120 + "\n")
#             f.write(f"RUN:  {run_name}\n")
#             f.write(f"FILE: {path.name}\n")
#             f.write(f"PATH: {path}\n")
#             f.write("=" * 120 + "\n")

#             f.write(f"Shape: {arr.shape}\n")
#             f.write(f"Dtype: {arr.dtype}\n")
#             f.write("Contents:\n")

#             f.write(
#                 np.array2string(
#                     arr,
#                     threshold=np.inf,
#                     max_line_width=200
#                 )
#             )

#             f.write("\n\n")

#         except Exception as e:
#             f.write("=" * 120 + "\n")
#             f.write(f"FILE: {path}\n")
#             f.write("=" * 120 + "\n")
#             f.write(
#                 f"ERROR: {type(e).__name__}: {e}\n\n"
#             )


# print(f"Done. Output written to: {OUTPUT_FILE}")

from pathlib import Path
from datetime import datetime
import json
import os

import numpy as np


# ======================================================================================
# CONFIG
# ======================================================================================

BASE_FOLDER = Path(
    "/scratch/m23csa510/cross_eval_runs/clean_eval"
)

OUTPUT_DIR = Path("extracted_results")

LOG_MAX_LINES = 60


# ======================================================================================
# HELPERS
# ======================================================================================

def format_timestamp(timestamp):
    """Convert Unix timestamp to readable local datetime."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def get_file_metadata(path):
    """
    Get useful file metadata.

    Linux does not always expose true file creation/birth time.
    If birth time exists, use it.
    Otherwise report modification time explicitly as the fallback.
    """
    stat = path.stat()

    metadata = {
        "size_bytes": stat.st_size,
        "modified_at": format_timestamp(stat.st_mtime),
    }

    # macOS / BSD and some filesystems expose st_birthtime
    if hasattr(stat, "st_birthtime"):
        metadata["timestamp"] = format_timestamp(stat.st_birthtime)
        metadata["timestamp_type"] = "Created/Birth time"
    else:
        # Do NOT use st_ctime as creation time on Linux:
        # it is inode metadata-change time.
        metadata["timestamp"] = format_timestamp(stat.st_mtime)
        metadata["timestamp_type"] = "Modified time (creation time unavailable)"

    return metadata


def write_file_header(out, run_dir, path):
    """Write consistent metadata/header for every extracted file."""
    metadata = get_file_metadata(path)

    out.write("-" * 120 + "\n")
    out.write(f"RUN:            {run_dir.name}\n")
    out.write(f"FILE:           {path.name}\n")
    out.write(f"FULL PATH:      {path.resolve()}\n")
    out.write(
        f"TIMESTAMP:      {metadata['timestamp']} "
        f"[{metadata['timestamp_type']}]\n"
    )
    out.write(f"LAST MODIFIED:  {metadata['modified_at']}\n")
    out.write(f"SIZE:           {metadata['size_bytes']} bytes\n")
    out.write("-" * 120 + "\n")


def write_run_header(out, run_dir):
    """Write a visual separator between evaluation/result folders."""
    out.write("\n\n")
    out.write("#" * 140 + "\n")
    out.write(f"RESULT FOLDER: {run_dir.name}\n")
    out.write(f"DIRECTORY:     {run_dir.resolve()}\n")
    out.write("#" * 140 + "\n\n")


# ======================================================================================
# FILE READERS
# ======================================================================================

def extract_npy(path, out):
    """Load and dump a .npy file."""
    arr = np.load(path, allow_pickle=True)

    out.write(f"Python type: {type(arr)}\n")

    if isinstance(arr, np.ndarray):
        out.write(f"Shape:       {arr.shape}\n")
        out.write(f"Dtype:       {arr.dtype}\n")

    out.write("\nCONTENTS:\n")

    if isinstance(arr, np.ndarray):
        out.write(
            np.array2string(
                arr,
                threshold=np.inf,
                max_line_width=200
            )
        )
    else:
        out.write(str(arr))

    out.write("\n")


def extract_log(path, out):
    """Write only the first LOG_MAX_LINES lines of a log file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = []

        for i, line in enumerate(f):
            if i >= LOG_MAX_LINES:
                break

            lines.append(line)

    out.write(
        f"CONTENTS: first {len(lines)} line(s), "
        f"maximum configured = {LOG_MAX_LINES}\n\n"
    )

    out.writelines(lines)

    if lines and not lines[-1].endswith("\n"):
        out.write("\n")


def extract_json(path, out):
    """
    Parse JSON and pretty-print it.
    Falls back to raw text if the file isn't valid JSON.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    try:
        data = json.loads(raw_text)

        out.write("CONTENTS:\n\n")
        out.write(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            )
        )
        out.write("\n")

    except json.JSONDecodeError as exc:
        out.write(
            f"WARNING: JSON parsing failed: {exc}\n"
            "Writing raw file contents instead.\n\n"
        )
        out.write(raw_text)

        if raw_text and not raw_text.endswith("\n"):
            out.write("\n")


def extract_csv(path, out):
    """
    Write the CSV as-is.

    Keeping CSV text intact is useful when sharing experiment results
    because headers and rows remain easy to read/copy.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    out.write("CONTENTS:\n\n")
    out.write(content)

    if content and not content.endswith("\n"):
        out.write("\n")


# ======================================================================================
# MAIN EXTRACTION
# ======================================================================================

def extract_clean_eval(base_folder, output_dir):
    base_folder = Path(base_folder)
    output_dir = Path(output_dir)

    if not base_folder.exists():
        raise FileNotFoundError(
            f"Base folder does not exist: {base_folder}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        ".npy": output_dir / "npy_contents.txt",
        ".log": output_dir / "log_contents.txt",
        ".json": output_dir / "json_contents.txt",
        ".csv": output_dir / "csv_contents.txt",
    }

    handlers = {
        ".npy": extract_npy,
        ".log": extract_log,
        ".json": extract_json,
        ".csv": extract_csv,
    }

    # Open all four aggregate output files once.
    outputs = {
        ext: open(path, "w", encoding="utf-8")
        for ext, path in output_paths.items()
    }

    counts = {
        ".npy": 0,
        ".log": 0,
        ".json": 0,
        ".csv": 0,
    }

    try:
        # Typical CleanEval layout:
        #
        # clean_eval/
        #   run_1/
        #       xxx.npy
        #       xxx.log
        #       xxx.json
        #       xxx.csv
        #   run_2/
        #       ...
        #
        # This recursively discovers result directories containing any
        # supported files, so it also works if nesting changes slightly.

        run_dirs = sorted({
            path.parent
            for path in base_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in handlers
        })

        print(f"Found {len(run_dirs)} result folder(s).\n")

        for run_dir in run_dirs:

            # Only files directly inside this result folder.
            # Nested directories are separately discovered as run_dirs.
            files_by_type = {
                ext: sorted(
                    path
                    for path in run_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() == ext
                )
                for ext in handlers
            }

            for ext, files in files_by_type.items():

                if not files:
                    continue

                out = outputs[ext]

                write_run_header(out, run_dir)

                for path in files:
                    try:
                        write_file_header(
                            out=out,
                            run_dir=run_dir,
                            path=path
                        )

                        handlers[ext](path, out)

                        counts[ext] += 1

                    except Exception as exc:
                        out.write(
                            f"\nERROR READING FILE:\n"
                            f"{type(exc).__name__}: {exc}\n"
                        )

                    out.write("\n")

    finally:
        for out in outputs.values():
            out.close()

    # ==================================================================================
    # SUMMARY
    # ==================================================================================

    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)

    for ext, count in counts.items():
        print(
            f"{ext:<6} : {count:>4} file(s) -> "
            f"{output_paths[ext]}"
        )


# ======================================================================================
# RUN
# ======================================================================================

if __name__ == "__main__":
    extract_clean_eval(
        base_folder=BASE_FOLDER,
        output_dir=OUTPUT_DIR
    )