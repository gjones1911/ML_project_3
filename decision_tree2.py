from G_d_tree import *
# from DisplayMethods import *
# from perfomance_tests import *
from user_input import *
import sys

# headers for breast cancer data
headers = ['Clump Thickness',                   # 0
           'Uniformity of Cell Size',           # 1
           'Uniformity of Cell Shape',          # 2
           'Marginal Adhesion',                 # 3
           'Single Epithelial Cell Size',       # 4
           'Bare Nuclei',                       # 5
           'Bland Chromatin',                   # 6
           'Normal Nucleoli',                   # 7
           'Mitoses',                           # 8
           'Class']                             # 9 note: class = 2 -> benign, = 4 -> malignant


use_headers = False

num_obs = 0
num_dim = 0

data_np = None

er_idx = 0

limit_type = 0

depth_limit = 2

imp_thrsh = .15

# load needed data file
if len(sys.argv) == 1:
    data_np = load_data_file('breast-cancer-wisconsin.dt')
    data_np = np.array(data_np, dtype=np.int)
    data_dim = data_np.shape
    num_obs = data_dim[0]
    num_dim = data_dim[1]
elif len(sys.argv) == 2:
    data_np = load_data_file(sys.argv[1])
    data_np = np.array(data_np, dtype=np.int)
    data_dim = data_np.shape
    num_obs = data_dim[0]
    num_dim = data_dim[1]
    headers = generate_col_headers(num_dim, verbose=False, alphabet=True)
elif len(sys.argv) > 2:
    data_np = load_data_file(sys.argv[1])
    data_np = np.array(data_np, dtype=np.int)
    data_dim = data_np.shape
    num_obs = data_dim[0]
    num_dim = data_dim[1]
    if use_headers:
        headers = generate_col_headers(num_dim, verbose=False, alphabet=True, head_arry=headers)
    else:
        headers = generate_col_headers(num_dim, verbose=False, alphabet=True, head_arry=sys.argv)

class_column = num_dim - 1

print('There are {:d} observations and there are {:d} attributes.'.format(num_obs, num_dim))

tree_file = 'best_tree.tree'

try:
    o_avg_err, o_depth_limit, o_tr_idx = read_d_tree(tree_file)
except FileNotFoundError:
    o_avg_err = 0
    o_depth_limit = 0
    o_tr_idx = 0


test_num = 1

error_type = ['entropy', 'gini', 'miss class']


# #################################################################################################################
# #########################     user input section     ############################################################
# #################################################################################################################

# get what type of tree limiter the user wishes to use
# 1 is depth_limit, 2 is impurity limit
limit_type, depth_limit, imp_thrsh = process_limit_type()

# get what type of impurty test
er_idx = process_error_type()

# #################################################################################################################
# #########################    Create randomized training, validation and test sets    ############################
# #########################    and find the best model with the set of limits          ############################
# #########################    and print results of testing and show modeling on       ############################
# #########################    testing set                                             ############################
# #################################################################################################################
accuracy, model = create_test_d_tree(data_np, er_idx, class_column,
                                                        limit_type, depth_limit, imp_thrsh, headers)


