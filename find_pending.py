# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import dataclasses
# import hashlib
# import json
# import re
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Tuple


# # -------------------------
# # Minimal replica of your spec + planner (no torch/ME imports)
# # -------------------------
# @dataclass(frozen=True)
# class PreprocSpec:
#     coord_norm_factor: float = 10.0
#     voxel_size: float = 0.05
#     z_scale: float = 1.0

#     intensity_mode: str = "as_is"
#     intensity_constant: float = 0.5
#     returns_mode: str = "as_is"
#     returns_k: int = 5
#     intensity_divisor: float = 65535.0

#     voxel_feat_mode: str = "sample_first"
#     mean_thin_p: float = 0.7
#     knn_k: int = 16
#     linearity_max_points: int = 50000
#     linearity_seed: int = 1337

#     tta_mode: str = "none"

#     patch_filter_mode: str = "none"
#     min_points: int = 1000
#     min_occ_vox: int = 1000

#     has_intensity_ref: bool = False

#     def to_id(self) -> str:
#         payload = json.dumps(dataclasses.asdict(self), sort_keys=True).encode("utf-8")
#         return hashlib.md5(payload).hexdigest()[:10]


# def make_run_plan(base: PreprocSpec, mode: str, max_runs: int) -> List[PreprocSpec]:
#     runs: List[PreprocSpec] = [base]

#     # "Kill switch" runs
#     runs.append(
#         dataclasses.replace(base, intensity_mode="zero", voxel_feat_mode="mean_all")
#     )
#     runs.append(
#         dataclasses.replace(
#             base, intensity_mode="quantile_match", voxel_feat_mode="mean_all"
#         )
#     )
#     runs.append(
#         dataclasses.replace(
#             base,
#             intensity_mode="proxy_linearity",
#             voxel_feat_mode="sample_max_linearity",
#         )
#     )
#     runs.append(dataclasses.replace(base, tta_mode="rot4", voxel_feat_mode="mean_all"))

#     intensity_pool = [
#         ("as_is", {}),
#         ("zero", {}),
#         ("constant", {"intensity_constant": 0.5}),
#         ("proxy_z", {}),
#         ("minmax_z", {}),
#         ("proxy_linearity", {}),
#         ("quantile_match", {}),
#     ]
#     returns_pool = [("as_is", {}), ("drop", {}), ("const1", {}), ("uniform", {})]
#     coord_pool = [0.8, 1.0, 1.2]
#     voxel_pool = [0.8, 1.0, 1.2]
#     zscale_pool = [0.8, 1.0, 1.2]
#     voxfeat_pool = [
#         ("sample_first", {}),
#         ("sample_random", {}),
#         ("sample_max_intensity", {}),
#         ("sample_max_linearity", {}),
#         ("mean_all", {}),
#         ("mean_thin", {"mean_thin_p": 0.7}),
#     ]
#     tta_pool = ["none", "rot4"]
#     filter_pool = [
#         ("none", {}),
#         ("min_points", {"min_points": 1000}),
#         ("min_occ_vox", {"min_occ_vox": 1000}),
#     ]

#     def add(r: PreprocSpec) -> None:
#         if len(runs) < max_runs:
#             runs.append(r)

#     if mode == "staged":
#         for im, kw in intensity_pool:
#             add(dataclasses.replace(base, intensity_mode=im, **kw))
#         for rm, kw in returns_pool:
#             add(dataclasses.replace(base, returns_mode=rm, **kw))
#         for m in coord_pool:
#             add(dataclasses.replace(base, coord_norm_factor=base.coord_norm_factor * m))
#         for m in voxel_pool:
#             add(dataclasses.replace(base, voxel_size=base.voxel_size * m))
#         for zs in zscale_pool:
#             add(dataclasses.replace(base, z_scale=zs))
#         for vm, kw in voxfeat_pool:
#             add(dataclasses.replace(base, voxel_feat_mode=vm, **kw))
#         for tm in tta_pool:
#             add(dataclasses.replace(base, tta_mode=tm))
#         for fm, kw in filter_pool:
#             add(dataclasses.replace(base, patch_filter_mode=fm, **kw))

#         combos = [
#             dataclasses.replace(
#                 base,
#                 intensity_mode="constant",
#                 intensity_constant=0.5,
#                 voxel_feat_mode="mean_all",
#                 tta_mode="rot4",
#             ),
#             dataclasses.replace(
#                 base,
#                 voxel_feat_mode="sample_max_linearity",
#                 voxel_size=base.voxel_size * 0.8,
#                 intensity_mode="constant",
#             ),
#             dataclasses.replace(
#                 base,
#                 z_scale=1.2,
#                 voxel_size=base.voxel_size * 0.8,
#                 voxel_feat_mode="mean_all",
#                 intensity_mode="constant",
#             ),
#         ]
#         for c in combos:
#             add(c)

#     elif mode == "full":
#         for im, imkw in intensity_pool:
#             for rm, rmkw in returns_pool:
#                 for cm in coord_pool:
#                     for vmult in voxel_pool:
#                         for zs in zscale_pool:
#                             for vfm, vfkw in voxfeat_pool:
#                                 for tm in tta_pool:
#                                     for fm, fmkw in filter_pool:
#                                         if len(runs) >= max_runs:
#                                             break
#                                         runs.append(
#                                             dataclasses.replace(
#                                                 base,
#                                                 intensity_mode=im,
#                                                 returns_mode=rm,
#                                                 coord_norm_factor=base.coord_norm_factor
#                                                 * cm,
#                                                 voxel_size=base.voxel_size * vmult,
#                                                 z_scale=zs,
#                                                 voxel_feat_mode=vfm,
#                                                 tta_mode=tm,
#                                                 patch_filter_mode=fm,
#                                                 **imkw,
#                                                 **rmkw,
#                                                 **vfkw,
#                                                 **fmkw,
#                                             )
#                                         )
#     else:
#         raise ValueError(f"Unknown mode: {mode}")

#     # de-dup (same as your code)
#     seen = set()
#     uniq: List[PreprocSpec] = []
#     for r in runs:
#         rid = r.to_id()
#         if rid in seen:
#             continue
#         uniq.append(r)
#         seen.add(rid)
#     return uniq[:max_runs]


# # -------------------------
# # Utilities to read existing run folders
# # -------------------------
# RUN_DIR_RE = re.compile(r"^run_(\d{3})_([0-9a-f]{10})$")


# def load_json(path: Path) -> Dict:
#     return json.loads(path.read_text())


# def is_valid_metrics(metrics_path: Path) -> bool:
#     if not metrics_path.exists():
#         return False
#     try:
#         obj = load_json(metrics_path)
#         # A "successful" run writes full/filtered keys
#         return (
#             isinstance(obj, dict)
#             and ("full" in obj)
#             and ("filtered" in obj)
#             and ("spec" in obj)
#         )
#     except Exception:
#         return False


# def main() -> None:
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--out_dir",
#         required=True,
#         help="Your sweep OUT_DIR (contains run_###_* folders)",
#     )
#     ap.add_argument("--mode", default="staged", choices=["staged", "full"])
#     ap.add_argument("--max_runs", type=int, default=120)
#     ap.add_argument("--include_slow_modes", action="store_true")
#     ap.add_argument(
#         "--write_files",
#         action="store_true",
#         help="Write pending_specs.json + snippet into out_dir",
#     )
#     args = ap.parse_args()

#     out_dir = Path(args.out_dir).resolve()
#     if not out_dir.exists():
#         raise SystemExit(f"out_dir not found: {out_dir}")

#     # 1) Load base spec from run_001_* (most robust because it includes intensity_constant median from ref JSON)
#     run1 = next(out_dir.glob("run_001_*/spec.json"), None)
#     if run1 is None:
#         raise SystemExit(
#             f"Could not find {out_dir}/run_001_*/spec.json. "
#             f"Need run_001 to reconstruct the exact base spec."
#         )
#     base_dict = load_json(run1)
#     base = PreprocSpec(**base_dict)

#     # 2) Recreate plan exactly like your job (plan=sweep -> make_run_plan + gating)
#     plan = make_run_plan(base, mode=args.mode, max_runs=args.max_runs)

#     # gating identical to your script
#     gated: List[PreprocSpec] = []
#     for s in plan:
#         if (s.intensity_mode == "quantile_match") and (not base.has_intensity_ref):
#             continue
#         if (not args.include_slow_modes) and (
#             (s.intensity_mode == "proxy_linearity")
#             or (s.voxel_feat_mode == "sample_max_linearity")
#         ):
#             continue
#         gated.append(s)
#     plan = gated[: args.max_runs]

#     expected_ids = [s.to_id() for s in plan]

#     # 3) Scan existing runs
#     done_ids = set()
#     incomplete_dirs: List[str] = []
#     unknown_dirs: List[str] = []

#     for p in sorted(out_dir.iterdir()):
#         if not p.is_dir():
#             continue
#         m = RUN_DIR_RE.match(p.name)
#         if not m:
#             continue

#         run_idx, run_id_from_name = m.group(1), m.group(2)
#         spec_path = p / "spec.json"
#         metrics_path = p / "metrics.json"

#         # If metrics.json exists + valid => done
#         if is_valid_metrics(metrics_path):
#             done_ids.add(run_id_from_name)
#             continue

#         # If folder exists but no valid metrics => incomplete (killed, crashed, etc.)
#         if spec_path.exists():
#             incomplete_dirs.append(p.name)
#         else:
#             unknown_dirs.append(p.name)

#     pending = [s for s in plan if s.to_id() not in done_ids]
#     pending_ids = [s.to_id() for s in pending]

#     print("==================================================")
#     print(f"[out_dir] {out_dir}")
#     print(
#         f"[plan] mode={args.mode} include_slow_modes={bool(args.include_slow_modes)} expected_runs={len(plan)}"
#     )
#     print(f"[done] {len(done_ids)}")
#     print(f"[pending] {len(pending)}")
#     print(
#         f"[incomplete_dirs] {len(incomplete_dirs)} (folders exist but metrics.json missing/invalid)"
#     )
#     print("==================================================")

#     if incomplete_dirs:
#         print("\nFolders that look started but not completed:")
#         for name in incomplete_dirs:
#             print(f"  - {name}")

#     print("\nPending run_ids (in planned order):")
#     for j, rid in enumerate(pending_ids, 1):
#         print(f"  {j:02d}) {rid}")

#     # Show compact spec deltas for sanity
#     def compact(s: PreprocSpec) -> str:
#         return (
#             f"intensity={s.intensity_mode}"
#             f", returns={s.returns_mode}"
#             f", coord={s.coord_norm_factor:g}"
#             f", vox={s.voxel_size:g}"
#             f", z={s.z_scale:g}"
#             f", vfeat={s.voxel_feat_mode}"
#             f", tta={s.tta_mode}"
#             f", filter={s.patch_filter_mode}"
#         )

#     print("\nPending specs (compact):")
#     for rid, s in zip(pending_ids, pending):
#         print(f"  - {rid}: {compact(s)}")

#     if args.write_files:
#         pending_specs_path = out_dir / "pending_specs.json"
#         pending_specs_path.write_text(
#             json.dumps([dataclasses.asdict(s) for s in pending], indent=2)
#         )

#         snippet_path = out_dir / "pending_plan_snippet.py"
#         snippet_path.write_text(
#             "# Auto-generated. Usage:\n"
#             "#   import json\n"
#             "#   from pending_plan_snippet import load_pending_specs\n"
#             "#   plan = load_pending_specs('pending_specs.json')\n"
#             "import json\n"
#             "from dataclasses import dataclass\n"
#             "from typing import List\n\n"
#             "@dataclass(frozen=True)\n"
#             "class PreprocSpec:\n"
#             "    coord_norm_factor: float\n"
#             "    voxel_size: float\n"
#             "    z_scale: float\n"
#             "    intensity_mode: str\n"
#             "    intensity_constant: float\n"
#             "    returns_mode: str\n"
#             "    returns_k: int\n"
#             "    intensity_divisor: float\n"
#             "    voxel_feat_mode: str\n"
#             "    mean_thin_p: float\n"
#             "    knn_k: int\n"
#             "    linearity_max_points: int\n"
#             "    linearity_seed: int\n"
#             "    tta_mode: str\n"
#             "    patch_filter_mode: str\n"
#             "    min_points: int\n"
#             "    min_occ_vox: int\n"
#             "    has_intensity_ref: bool\n\n"
#             "def load_pending_specs(path: str) -> List[PreprocSpec]:\n"
#             "    data = json.load(open(path, 'r'))\n"
#             "    return [PreprocSpec(**d) for d in data]\n"
#         )

#         print("\n[saved]")
#         print(f"  - {pending_specs_path}")
#         print(f"  - {snippet_path}")
#         print("You can now hard-run only these specs using pending_specs.json.")


# if __name__ == "__main__":
#     main()

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ==========================================
# 1. EXACT COPY OF DATA CLASSES FROM ORIGINAL
# ==========================================


@dataclass(frozen=True)
class PreprocSpec:
    coord_norm_factor: float = 10.0
    voxel_size: float = 0.05
    z_scale: float = 1.0
    intensity_mode: str = "as_is"
    intensity_constant: float = 0.5
    returns_mode: str = "as_is"
    returns_k: int = 5
    intensity_divisor: float = 65535.0
    voxel_feat_mode: str = "sample_first"
    mean_thin_p: float = 0.7
    knn_k: int = 16
    linearity_max_points: int = 50000
    linearity_seed: int = 1337
    tta_mode: str = "none"
    patch_filter_mode: str = "none"
    min_points: int = 1000
    min_occ_vox: int = 1000
    has_intensity_ref: bool = False

    def to_id(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.md5(payload).hexdigest()[:10]


# ==========================================
# 2. EXACT COPY OF PLAN GENERATOR
# ==========================================


def make_run_plan(base: PreprocSpec, mode: str, max_runs: int) -> List[PreprocSpec]:
    runs: List[PreprocSpec] = [base]

    # Kill Switch / Science-first runs
    runs.append(
        dataclasses.replace(base, intensity_mode="zero", voxel_feat_mode="mean_all")
    )
    runs.append(
        dataclasses.replace(
            base, intensity_mode="quantile_match", voxel_feat_mode="mean_all"
        )
    )
    runs.append(
        dataclasses.replace(
            base,
            intensity_mode="proxy_linearity",
            voxel_feat_mode="sample_max_linearity",
        )
    )
    runs.append(dataclasses.replace(base, tta_mode="rot4", voxel_feat_mode="mean_all"))

    intensity_pool = [
        ("as_is", {}),
        ("zero", {}),
        ("constant", {"intensity_constant": 0.5}),
        ("proxy_z", {}),
        ("minmax_z", {}),
        ("proxy_linearity", {}),
        ("quantile_match", {}),
    ]
    returns_pool = [("as_is", {}), ("drop", {}), ("const1", {}), ("uniform", {})]
    coord_pool = [0.8, 1.0, 1.2]
    voxel_pool = [0.8, 1.0, 1.2]
    zscale_pool = [0.8, 1.0, 1.2]
    voxfeat_pool = [
        ("sample_first", {}),
        ("sample_random", {}),
        ("sample_max_intensity", {}),
        ("sample_max_linearity", {}),
        ("mean_all", {}),
        ("mean_thin", {"mean_thin_p": 0.7}),
    ]
    tta_pool = ["none", "rot4"]
    filter_pool = [
        ("none", {}),
        ("min_points", {"min_points": 1000}),
        ("min_occ_vox", {"min_occ_vox": 1000}),
    ]

    def add(r: PreprocSpec):
        if len(runs) < max_runs:
            runs.append(r)

    if mode == "staged":
        for im, kw in intensity_pool:
            add(dataclasses.replace(base, intensity_mode=im, **kw))
        for rm, kw in returns_pool:
            add(dataclasses.replace(base, returns_mode=rm, **kw))
        for m in coord_pool:
            add(dataclasses.replace(base, coord_norm_factor=base.coord_norm_factor * m))
        for m in voxel_pool:
            add(dataclasses.replace(base, voxel_size=base.voxel_size * m))
        for zs in zscale_pool:
            add(dataclasses.replace(base, z_scale=zs))
        for vm, kw in voxfeat_pool:
            add(dataclasses.replace(base, voxel_feat_mode=vm, **kw))
        for tm in tta_pool:
            add(dataclasses.replace(base, tta_mode=tm))
        for fm, kw in filter_pool:
            add(dataclasses.replace(base, patch_filter_mode=fm, **kw))

        combos = [
            dataclasses.replace(
                base,
                intensity_mode="constant",
                intensity_constant=0.5,
                voxel_feat_mode="mean_all",
                tta_mode="rot4",
            ),
            dataclasses.replace(
                base,
                voxel_feat_mode="sample_max_linearity",
                voxel_size=base.voxel_size * 0.8,
                intensity_mode="constant",
            ),
            dataclasses.replace(
                base,
                z_scale=1.2,
                voxel_size=base.voxel_size * 0.8,
                voxel_feat_mode="mean_all",
                intensity_mode="constant",
            ),
        ]
        for c in combos:
            add(c)

    # Dedup
    seen = set()
    uniq = []
    for r in runs:
        rid = r.to_id()
        if rid in seen:
            continue
        uniq.append(r)
        seen.add(rid)
    return uniq[:max_runs]


# ==========================================
# 3. MAIN RECOVERY LOGIC
# ==========================================


def main():
    # PATH FROM YOUR CONTEXT
    results_dir = Path(
        "/scratch/m23csa510/dales_eval_runs/dales_common_all_20251230_175739_24022"
    )

    # 1. Recover Base Spec from Run 001
    # We do this to get the EXACT intensity_constant (0.6637...) used in the original run
    # otherwise hashes won't match if we use the default 0.5
    run_001 = list(results_dir.glob("run_001_*"))[0]
    spec_json = run_001 / "spec.json"
    print(f"Loading base configuration from: {spec_json}")

    with open(spec_json, "r") as f:
        data = json.load(f)
        # Reconstruct the PreprocSpec object from JSON data
        base_spec = PreprocSpec(**data)

    # 2. Re-generate the full plan
    # Your SLURM script used mode="staged", max_runs=120
    # Note: include_slow_modes was True in your SLURM script, so we don't filter anything out here.
    full_plan = make_run_plan(base_spec, mode="staged", max_runs=120)

    print(f"Reconstructed original plan size: {len(full_plan)} runs")

    # 3. Find executed runs in directory
    executed_ids = set()
    for run_dir in results_dir.glob("run_*"):
        if not run_dir.is_dir():
            continue
        # Verify it actually has a spec.json (implies it started)
        if (run_dir / "spec.json").exists():
            # Extract ID from folder name (run_XXX_ID)
            parts = run_dir.name.split("_")
            if len(parts) >= 3:
                rid = parts[2]
                executed_ids.add(rid)

    print(f"Found {len(executed_ids)} executed runs on disk.")

    # 4. Filter for pending
    pending_specs = []
    for spec in full_plan:
        if spec.to_id() not in executed_ids:
            pending_specs.append(spec)

    print(f"Pending runs found: {len(pending_specs)}")
    print("-" * 60)

    # 5. Output Code Block
    print("COPY THE CODE BELOW INTO YOUR RESCUE SCRIPT:")
    print("-" * 60)
    print("pending_plan = [")
    for spec in pending_specs:
        print(f"    {spec},")
    print("]")
    print("-" * 60)


if __name__ == "__main__":
    main()
