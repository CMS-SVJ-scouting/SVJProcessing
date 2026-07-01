import os
import subprocess

from utils.Logger import *


def get_temporary_directory():
    if "cmslpc" in os.environ["HOSTNAME"]:
        temporary_directory = f"/uscmst1b_scratch/lpc1/3DayLifetime/{os.environ['USER']}"
    elif "t3ui" in os.environ["HOSTNAME"]:
        temporary_directory = f"/tmp/{os.environ['USER']}"
    elif "portal1" in os.environ["HOSTNAME"] or "bms" in os.environ["HOSTNAME"] or "deepthought" in os.environ["HOSTNAME"]:
        temporary_directory = f"/tmp/{os.environ['USER']}"
    else:
        log.critical(f"Unknown hostname {os.environ['HOSTNAME']}")
        exit(1)

    return temporary_directory


def run_bash_command(bash_command):
    """Returns the output of a bash command.

    Args:
        bash_command (str)

    Returns:
        str
    """

    return subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE).stdout.read().decode("utf-8")[:-1]


def test_command(bash_command):
    print("Testing:")
    print(bash_command)

    print("\nRunning...")
    result = subprocess.run(bash_command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

    print("Done!\n")

