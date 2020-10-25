import random
import numpy as np


sql = "explain analyse select * from {} where {};\n"
template = "column{} <= {}"
and_str = " and "


def p_dim(dim):
    sum = 0
    for i in range(1, dim + 1):
        sum += i * i
    return [float(i * i) / float(sum) for i in range(1, dim + 1)]


def get_grides(splilt_num, low_col, high_col):
    return [np.linspace(low, high, splilt_num) for low, high in zip(low_col, high_col)]


def range_data_generator(tablename, dim, datasize, low_col, high_col):
    grids_idx = [i for i in range(dim)]
    grides = get_grides(dim + 1, low_col, high_col)
    p_right = p_dim(dim)
    # p_left = p_right[::-1]
    query = []
    num_generate = 0
    query_range = []
    while num_generate < datasize:
        # left_idx = np.random.choice(grids_idx, size=dim, p=p_left)
        right_idx = np.random.choice(grids_idx, size=dim, p=p_right)
        # left = [
        #     int(random.uniform(grides[dim_idx][grid_idx], grides[dim_idx][grid_idx + 1]))
        #     for dim_idx, grid_idx in enumerate(left_idx)
        # ]
        right = [
            int(random.uniform(grides[dim_idx][grid_idx], grides[dim_idx][grid_idx + 1]))
            for dim_idx, grid_idx in enumerate(right_idx)
        ]
        # predicate = [template.format(idx, lef, idx, ri) for idx, (lef, ri) in enumerate(zip(left, right)) if lef < ri]
        range_tmp = [ri for ri in right]

        if len(range_tmp) != dim:
            continue
        num_generate += 1
        query_range.append(np.array(range_tmp).ravel())
        predicate = [template.format(idx, ri) for idx, ri in enumerate(range_tmp)]
        predicate = and_str.join(predicate)
        query.append(sql.format(tablename, predicate))
    return query, query_range


if __name__ == "__main__":
    low_col = [1, 2, 3, 4]
    high_col = [100, 200, 300, 400]
    querys, query_range = range_data_generator("cover", 4, 1000, low_col, high_col)
    with open("data/grid_cdf_cover_workload.txt", "w") as f:
        f.writelines(querys)
    np.save("data/cdf_test.npy", np.array(query_range))
