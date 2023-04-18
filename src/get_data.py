import argparse
import os
import requests
from multiprocessing import Process

import mido
import torch
from tqdm import tqdm


def file_counter(directory, ext):
    """
    Returns function that counts files in directory with given extension(s).
    """
    def func():
        return len([f for f in os.listdir(directory) if f.endswith(ext)])
    return func


def multiprocess(target, num_jobs, args, total, progress, desc=""):
    """
    Start concurrent processes.
    :param target: Worker function.
    :param num_jobs: Number of processes to start.
    :param args: List of arguments for each process.
    :param total: Total number of items to process.
    :param progress: Function that returns how many things currently done.
    :param desc: Description for progress bar.
    """
    procs = []
    for i in range(num_jobs):
        p = Process(target=target, args=args[i])
        p.start()
        procs.append(p)

    pbar = tqdm(total=total, desc=desc)
    while any(p.is_alive() for p in procs):
        num_done = progress()
        pbar.update(num_done - pbar.n)
    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="data")
    parser.add_argument("-j", type=int, default=8)
    args = parser.parse_args()

    # TODO: Actual stuff


if __name__ == "__main__":
    main()
