#!/usr/bin/env python3
import h5py
import argparse
import sys

def print_tree_structure(file_path):
    """打印 HDF5 文件的树状结构"""
    def print_node(node, prefix="", is_last=False):
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{node.name.split('/')[-1]}")
        new_prefix = prefix + ("    " if is_last else "│   ")

        if isinstance(node, h5py.Group):
            children = list(node.items())
            for i, (_, child) in enumerate(children):
                print_node(child, new_prefix, i == len(children) - 1)
        elif isinstance(node, h5py.Dataset):
            print(f"{new_prefix}Shape: {node.shape}")

    try:
        with h5py.File(file_path, 'r') as f:
            print(f.name)
            children = list(f.items())
            for i, (_, child) in enumerate(children):
                print_node(child, "", i == len(children) - 1)
    except Exception as e:
        print(f"Error opening file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print HDF5 file structure as a tree.")
    parser.add_argument("-s", help="Path to the HDF5 file (.h5)", default="out.h5")
    args = parser.parse_args()

    print_tree_structure(args.s)