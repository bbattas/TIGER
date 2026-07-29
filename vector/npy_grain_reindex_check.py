import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Check unique grain values for npy version of vector data. ' \
                                'If the index begins from 0, shift to 1 minimum value.')
parser.add_argument('--input', '-i', required=True, help='Path to the input .npy file')
args = parser.parse_args()

filename = args.input
print(f'For {filename}')
data = np.load(filename).astype(np.int32)

# Check before
print(f"Before - Min: {data.min()}, Max: {data.max()}, Unique count: {len(np.unique(data[0]))}")

if data.min() == 0:
    # Shift all grain IDs up by 1 so they start from 1
    data = data + 1

    # Check after
    print(f"After  - Min: {data.min()}, Max: {data.max()}, Unique count: {len(np.unique(data[0]))}")

    # Save reindexed file
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_reindexed{ext}"
    np.save(output_filename, data)
    print(f"Saved reindexed file: {output_filename}.")
    print(" ")

