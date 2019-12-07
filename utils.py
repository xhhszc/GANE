import linecache
import numpy as np


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# Get batch data from training set
def get_batch_data(file, index, size):  # 1,5->1,2,3,4,5
    user = []
    item_real = []
    item_fake = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        line = line.strip()
        line = line.split()
        user.append(int(line[0]))
        item_real.append(int(line[1]))
        item_fake.append(int(line[2]))
    return user, item_real, item_fake


