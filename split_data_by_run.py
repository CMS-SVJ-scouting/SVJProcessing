import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REDIRECTOR = "root://cmsdcache-kit-disk.gridka.de"
BASE = "/store/user/mgaisdor/SVJScouting_ntuples/Data/2018_v2"
MAX_WORKERS = 8

# build GUID → path map
result = subprocess.check_output(
    ["xrdfs", REDIRECTOR, "ls", BASE],
    text=True
)

guid_map = {}
for line in result.splitlines():
    if "PFNano_" in line:
        guid = Path(line).stem.replace("PFNano_", "")
        guid_map[guid] = line

# create destination directories first
for era in ["A", "B", "C", "D"]:
    dst_dir = f"{BASE}/Run2018{era}"
    print(f"Creating directory {dst_dir}")
    subprocess.run(["xrdfs", REDIRECTOR, "mkdir", "-p", dst_dir], check=True)

# collect all moves
moves = []
for era in ["A", "B", "C", "D"]:
    with open(f"tmp/2018{era}.txt") as f:
        for raw in f:
            raw = raw.strip()
            guid = Path(raw).stem
            if guid in guid_map:
                src = guid_map[guid]
                dst = f"{BASE}/Run2018{era}/{Path(src).name}"
                moves.append((src, dst))
            else:
                print(f"Missing: {guid}")

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

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(copy_file, src, dst): (src, dst) for src, dst in moves}
    for future in as_completed(futures):
        src, dst = futures[future]
        try:
            future.result()
        except subprocess.CalledProcessError as e:
            print(f"Failed: {src} -> {dst}: {e}")