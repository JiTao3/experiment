from operator import le
import random
import numpy as np
from numpy.core.defchararray import lower
from numpy.core.function_base import linspace


"""
explain analyse select * from cover where column1 <= 2935 and column2 <= 39 and column3 <= 57 and column4 <= 509 and
column5 <= 446 and column6 <= 5647 and column7 <= 252 and column8 <= 34 and column9 <= 235 and column10 <= 5197;
"""
"""
explain analyse select * from cover where (column6 >= 3924 and column6 <= 4874) and (column8 >=8 and column8 <=140);
"""

sql = "explain analyse select * from {} where {};\n"
template = "(column{} >= {} and column{} <= {})"
and_str = " and "


def p_dim(dim):
    sum = 0
    for i in range(1, dim + 1):
        sum += i
    return [float(i) / float(sum) for i in range(1, dim + 1)]


def get_grides(splilt_num, low_col, high_col):
    return [np.linspace(low, high, splilt_num) for low, high in zip(low_col, high_col)]


def range_data_generator(tablename, dim, datasize, low_col, high_col):
    grids_idx = [i for i in range(dim)]
    grides = get_grides(dim + 1, low_col, high_col)
    p_right = p_dim(dim)
    p_left = p_right[::-1]
    query = []
    num_generate = 0
    query_range = []
    while num_generate <= datasize:
        left_idx = np.random.choice(grids_idx, size=dim, p=p_left)
        right_idx = np.random.choice(grids_idx, size=dim, p=p_right)
        left = [
            int(random.uniform(grides[dim_idx][grid_idx], grides[dim_idx][grid_idx + 1]))
            for dim_idx, grid_idx in enumerate(left_idx)
        ]
        right = [
            int(random.uniform(grides[dim_idx][grid_idx], grides[dim_idx][grid_idx + 1]))
            for dim_idx, grid_idx in enumerate(right_idx)
        ]
        # predicate = [template.format(idx, lef, idx, ri) for idx, (lef, ri) in enumerate(zip(left, right)) if lef < ri]
        range_tmp = [(lef, ri) for lef, ri in zip(left, right) if lef < ri]

        if len(range_tmp) != dim:
            continue
        num_generate += 1
        query_range.append(range_tmp)
        predicate = [template.format(idx, lef, idx, ri) for idx, (lef, ri) in enumerate(range_tmp)]
        predicate = and_str.join(predicate)
        query.append(sql.format(tablename, predicate))
    return query, query_range


if __name__ == "__main__":
    low_col = [1, 2, 3, 4]
    high_col = [100, 200, 300, 400]
    querys, query_range = range_data_generator("cover", 4, 1000, low_col, high_col)
    with open("data/grid_range_cover_workload.txt", "w") as f:
        f.writelines(querys)
    np.save('data/test.npy', np.array(query_range))
