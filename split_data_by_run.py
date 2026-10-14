"""Copy data ntuples into per-era subdirectories (Run<year><era>/).

Ntuples are named PFNano_<GUID>.root after their parent RAW file, so each is
matched to its era through per-era lists of RAW file LFNs from DAS, one per
era in tmp/<year><era>.txt, e.g. for 2016:

    for era in B C D E F G H; do
      for ds in $(dasgoclient -query="dataset=/ScoutingPFHT/Run2016${era}*/RAW"); do
        dasgoclient -query="file dataset=${ds}"
      done > tmp/2016${era}.txt
    done

Usage:
    python split_data_by_run.py -y 2016 -e B C D E F G H --dry-run
    python split_data_by_run.py -y 2016 -e B C D E F G H
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REDIRECTOR = "root://cmsdcache-kit-disk.gridka.de"
BASE_TEMPLATE = "/store/user/mgaisdor/SVJScouting_ntuples/Data/{year}_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", required=True, help="e.g. 2016")
    parser.add_argument("-e", "--eras", nargs="+", required=True, help="e.g. B C D E F G H")
    parser.add_argument("-b", "--base", default=None, help="Input directory, default: " + BASE_TEMPLATE)
    parser.add_argument("-l", "--lists-dir", default="tmp", help="Directory with the <year><era>.txt DAS lists")
    parser.add_argument("-j", "--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Only report the matching, copy nothing")
    return parser.parse_args()


def copy_file(src, dst):
    # Check if destination file already exists
    result = subprocess.run(
        ["xrdfs", REDIRECTOR, "stat", dst],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"Skipping {dst} (already exists)")
        return src, dst

    print(f"Copying {src} -> {dst}")
    subprocess.run(["xrdcp", f"{REDIRECTOR}/{src}", f"{REDIRECTOR}/{dst}"], check=True)
    return src, dst


def main():
    args = parse_args()
    base = args.base or BASE_TEMPLATE.format(year=args.year)

    # build GUID → path map
    result = subprocess.check_output(
        ["xrdfs", REDIRECTOR, "ls", base],
        text=True
    )

    guid_map = {}
    for line in result.splitlines():
        if "PFNano_" in line:
            guid = Path(line).stem.replace("PFNano_", "")
            guid_map[guid] = line

    # collect all moves
    moves = []
    matched = set()
    for era in args.eras:
        n_listed, n_missing = 0, 0
        with open(f"{args.lists_dir}/{args.year}{era}.txt") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                n_listed += 1
                guid = Path(raw).stem
                if guid in guid_map:
                    src = guid_map[guid]
                    dst = f"{base}/Run{args.year}{era}/{Path(src).name}"
                    moves.append((src, dst))
                    matched.add(guid)
                else:
                    n_missing += 1
                    print(f"Missing: {guid} ({args.year}{era})")
        print(f"Run{args.year}{era}: {n_listed - n_missing}/{n_listed} RAW files have an ntuple")

    unmatched = sorted(set(guid_map) - matched)
    for guid in unmatched:
        print(f"Not in any era list: {guid_map[guid]}")
    print(f"Total: {len(moves)} ntuples to copy, {len(unmatched)} of {len(guid_map)} ntuples not in any era list")

    if args.dry_run:
        return

    # create destination directories first
    for era in args.eras:
        dst_dir = f"{base}/Run{args.year}{era}"
        print(f"Creating directory {dst_dir}")
        subprocess.run(["xrdfs", REDIRECTOR, "mkdir", "-p", dst_dir], check=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(copy_file, src, dst): (src, dst) for src, dst in moves}
        for future in as_completed(futures):
            src, dst = futures[future]
            try:
                future.result()
            except subprocess.CalledProcessError as e:
                print(f"Failed: {src} -> {dst}: {e}")


if __name__ == "__main__":
    main()
