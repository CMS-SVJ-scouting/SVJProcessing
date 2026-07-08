"""Merge (hadd) the small per-job signal ntuple files into fewer, larger files.

Motivation: the Lund-plane reweighting distortion systematic (see
LundReweighting/lund_utils/LundReweighter.py's get_all_weights()) needs at least
1000 (ideally >=5000) jets within a single coffea processing chunk to be computed
at all. Since coffea's chunking never spans multiple input files (a chunk/WorkItem
is always tied to exactly one file), and these raw per-job ntuples only have on
the order of ~1000 events each, the distortion systematic is silently skipped for
almost every chunk today. This script merges FILES_PER_BATCH raw files at a time
into one larger file, so each merged file comfortably clears that jet threshold
on its own.

Batch size is intentionally moderate (not one giant file per dataset): merging
everything into a single very large file risks hitting the same ROOT basket-size
limit this framework already ran into once before (see git commit 0c9a97e /
its revert 0d571ae, about a 2GB basket size limit corrupting a branch).

Output layout: root://<redirector>/<...>/MC/<year>_v5/merged_signals/<signal_model>/merged_partNNNN.root

Usage:
    # dry run (default) - just prints what would be done, touches nothing
    python merge_signal_files.py

    # actually run the merge
    python merge_signal_files.py --execute

    # tune batch size / concurrency / year / subset of signal models
    python merge_signal_files.py --execute --files-per-batch 12 --n-workers 4 --year 2018 --signal-model s-channel_mMed-500_mDark-20_rinv-0.3
"""

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from dataset_configs.s_channel_scouting_dataset_paths import signal_models

REDIRECTOR = "root://cmsdcache-kit-disk.gridka.de:1094/"
BASE_PATH_TEMPLATE = "/store/user/mgaisdor/SVJScouting_ntuples/MC/{year}_v5/{signal_model}/"
MERGED_SUBDIR_TEMPLATE = "/store/user/mgaisdor/SVJScouting_ntuples/MC/{year}_v5/merged_signals/{signal_model}/"

DEFAULT_FILES_PER_BATCH = 10  # ~1000 raw events/file -> ~10,000 events/merged file;
                               # calibrate against the actual jets/event ratio for your
                               # selection before committing to a full production run.


def list_root_files(redirector, remote_dir):
    """List .root files in a remote directory via xrdfs. Returns sorted file names (not full paths)."""
    result = subprocess.run(
        ["xrdfs", redirector, "ls", remote_dir],
        capture_output=True, text=True, check=True,
    )
    files = [
        os.path.basename(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().endswith(".root")
    ]
    return sorted(files)


def ensure_remote_dir(redirector, remote_dir):
    subprocess.run(["xrdfs", redirector, "mkdir", "-p", remote_dir], check=False)


def remote_file_exists(redirector, remote_path):
    result = subprocess.run(
        ["xrdfs", redirector, "stat", remote_path],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def batch(iterable, size):
    """Yield consecutive slices of `iterable` of length `size`, except the last
    slice absorbs any remainder (so it's between size and 2*size-1 long) rather
    than being left as its own small, possibly under-threshold, trailing batch.
    If there are fewer than `size` items total, yields them all as one batch.
    """
    n = len(iterable)
    if n == 0:
        return
    n_full_batches = n // size
    if n_full_batches == 0:
        yield iterable[:]
        return
    remainder = n % size
    for i in range(n_full_batches):
        start = i * size
        end = start + size
        if i == n_full_batches - 1:
            end += remainder
        yield iterable[start:end]


def merge_one_batch(args):
    (redirector, source_dir, merged_dir, batch_files, batch_index, dry_run) = args

    output_name = f"merged_part{batch_index}.root"
    output_remote_path = merged_dir + output_name

    if not dry_run and remote_file_exists(redirector, output_remote_path):
        return f"[SKIP] {output_remote_path} already exists"

    source_paths = [f"{redirector}{source_dir}{fname}" for fname in batch_files]

    if dry_run:
        return (
            f"[DRY RUN] would hadd {len(source_paths)} files -> {output_remote_path}\n"
            f"           first: {source_paths[0]}\n"
            f"           last:  {source_paths[-1]}"
        )

    local_tmp = f"/tmp/{os.environ.get('USER', 'user')}/merge_signal_files/{output_name}"
    os.makedirs(os.path.dirname(local_tmp), exist_ok=True)

    try:
        # -f: overwrite local tmp if present (shouldn't be, but be safe on retries)
        hadd_cmd = ["hadd", "-f", local_tmp] + source_paths
        result = subprocess.run(hadd_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"[FAIL] hadd failed for batch {batch_index}: {result.stderr[-2000:]}"

        xrdcp_cmd = ["xrdcp", "-f", local_tmp, f"{redirector}{output_remote_path}"]
        result = subprocess.run(xrdcp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"[FAIL] xrdcp failed for batch {batch_index}: {result.stderr[-2000:]}"

        return f"[OK] {output_remote_path} ({len(batch_files)} files merged)"
    finally:
        if os.path.exists(local_tmp):
            os.remove(local_tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", default="2018", help="Dataset year (default: 2018)")
    parser.add_argument("--files-per-batch", type=int, default=DEFAULT_FILES_PER_BATCH,
                         help=f"Number of raw files to hadd per merged output file (default: {DEFAULT_FILES_PER_BATCH})")
    parser.add_argument("--n-workers", type=int, default=4,
                         help="Number of concurrent hadd+xrdcp operations (default: 4, be considerate of shared resources)")
    parser.add_argument("--signal-model", action="append", default=None,
                         help="Restrict to one or more specific signal models (repeatable). Default: all signal_models.")
    parser.add_argument("--execute", action="store_true",
                         help="Actually perform the merge. Without this flag, only prints what would be done.")
    args = parser.parse_args()

    dry_run = not args.execute
    models = args.signal_model if args.signal_model else signal_models

    if dry_run:
        print("=== DRY RUN: no files will be read, written, or created. Pass --execute to actually run. ===\n")

    n_ok, n_fail, n_skip = 0, 0, 0

    # One pool for the whole run (avoids repeated worker startup cost), but jobs
    # are submitted and drained one signal model at a time so each hypothesis
    # gets its own clean progress bar instead of interleaved output.
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for signal_model in models:
            source_dir = BASE_PATH_TEMPLATE.format(year=args.year, signal_model=signal_model)
            merged_dir = MERGED_SUBDIR_TEMPLATE.format(year=args.year, signal_model=signal_model)

            try:
                files = list_root_files(REDIRECTOR, source_dir)
            except subprocess.CalledProcessError as e:
                tqdm.write(f"[ERROR] could not list {source_dir}: {e.stderr}")
                continue

            if not files:
                tqdm.write(f"[WARN] no .root files found in {source_dir}, skipping")
                continue

            if not dry_run:
                ensure_remote_dir(REDIRECTOR, merged_dir)

            jobs = [
                (REDIRECTOR, source_dir, merged_dir, file_batch, batch_index, dry_run)
                for batch_index, file_batch in enumerate(batch(files, args.files_per_batch))
            ]

            if dry_run:
                # Dry run is instant (no real I/O), so just show one example batch
                # per dataset rather than animating a bar over it.
                print(f"{signal_model}: {len(files)} raw files -> {len(jobs)} merged files "
                      f"({args.files_per_batch} files/batch)")
                print(merge_one_batch(jobs[0]))
                continue

            futures = {pool.submit(merge_one_batch, job): job for job in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc=signal_model, unit="batch"):
                msg = future.result()
                if msg.startswith("[OK]"):
                    n_ok += 1
                elif msg.startswith("[SKIP]"):
                    n_skip += 1
                else:
                    n_fail += 1
                    tqdm.write(msg)

    if dry_run:
        print("\n(showed 1 example batch per dataset; pass --execute to actually run all batches)")
        return

    print(f"\nDone. {n_ok} merged, {n_skip} skipped (already existed), {n_fail} failed.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
