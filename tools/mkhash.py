#!/usr/bin/env python

import argparse
import os.path
import sys
#from Crypto.Hash import SHA256
from hashlib import sha256
from pathlib import Path
from typing import List, Iterable


def get_paths(paths_str: str, basedir) -> List[Path]:
    if paths_str:
        paths = [Path(basedir, p_str) for p_str in paths_str]
    else:
        paths = []
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ignore', metavar="IGNORE PATTERN", action='append', required=False, type=str, default=[])
    parser.add_argument('--home', metavar='HOME DIR', required=False, type=str)
    parser.add_argument('-o', '--output', metavar='FILE', required=False, type=str)
    parser.add_argument('path', metavar='PATH', nargs='+', type=str)

    args = parser.parse_args()
    ignore_patterns = args.ignore
    home_dir = Path(args.home) if args.home else Path('.')
    paths = get_paths(args.path, basedir=home_dir)
    output = args.output if args.output else sys.stdout

    files_to_checksum = get_files_to_checksum(paths, ignore_patterns)

    output_file = open(output, 'w') if output != sys.stdout else output

    for file in files_to_checksum:
        file_checksum = hash_file(file)
        print(f"{file}, {file_checksum}", file=output_file)

    output_file.close()


def get_files_to_checksum(paths: Iterable[Path], ignore_patterns):
    files_to_checksum = []
    for path in filter(lambda p: not is_file_ignored(ignore_patterns, p), paths):
        files_in_dir = [path] if path.is_file() else []
        for current_path, dirs, files in os.walk(path):
            fs = [Path(current_path, f) for f in files]
            fs = list(filter(lambda f: not is_file_ignored(ignore_patterns, f), fs))
            files_in_dir.extend(fs)

        files_to_checksum.extend(files_in_dir)

    return files_to_checksum


def hash_file(file: Path, buffer_size=64*1024):
    done = False
    checksum = sha256()
    with open(file, 'rb', buffering=0) as bin_file:
        while not done:
            data = bin_file.read(buffer_size)
            if len(data)>0:
                checksum.update(data)
            else:
                done = True

    return checksum.hexdigest()


def is_file_ignored(ignore_patterns, file):
    matches = [p in file.as_posix() for p in ignore_patterns]
    return any(matches)


if __name__ == '__main__':
    main()
