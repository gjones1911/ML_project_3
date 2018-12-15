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
#er_idx = 0

psum = 0

psum_dic = {2: 0, 8: 0, 17: 0, 33: 0}
best_err = {2: 0, 8: 0, 17: 0, 33: 0}
err_avg = {2: 0, 8: 0, 17: 0, 33: 0}

tr_dic = {2: None, 8: None, 17: None, 33: None}
train_dic = {2: None, 8: None, 17: None, 33: None}
ts_dic = {2: None, 8: None, 17: None, 33: None}
val_dic = {2: None, 8: None, 17: None, 33: None}

model_dic = {2: None, 8: None, 17: None, 33: None}

tr, ts, vs = list(), list(), list()


#imp_thrsh = list()
#depth_limit = list()
#limit_type = -2

# #################################################################################################################
# #########################     user input section     ############################################################
# #################################################################################################################

# ask the user if they want to just run one simple run with no depth limit, and gini impurity test
# has optional options for depth limit(a single number), impurity threshold(a single float)
#auto_run = process_yes_no(depth_limit=2)


'''
if len(auto_run) == 3:
    if auto_run[2] == 'd':
        depth_limit = auto_run[1]
        imp_thrsh = None
    elif auto_run[2] == 'i':
        depth_limit = None
        imp_thrsh = auto_run[1]
    auto_run = auto_run[0]
else:
'''

#depth_limit = None
#imp_thrsh = None
auto_run = False


if not auto_run:
    # get the number of test runs the user wishes to run for ba
    #test_num = process_num_input()

    #test_num = 1

    # get what type of tree limiter the user wishes to use
    # 1 is depth_limit, 2 is impurity limit
    limit_type, depth_limit, imp_thrsh = process_limit_type()

    # get what type of impurty test
    er_idx = process_error_type()


b_avg_accuracy = 0
b_single_avg = 0
b_avg_d = 0
b_s_d = 0
acc = 0
bad, bsd = list(), list()
cnt = 0
lim = 0

#create_test_d_tree2(data_np, test_num, error_type, er_idx, class_column, limit_type, depth_limit,
#                       imp_thrsh, headers):


accuracy, b_avg_d, b_sing_d, model = create_test_d_tree2(data_np, test_num, error_type, er_idx, class_column,
                                                         limit_type, depth_limit, imp_thrsh, headers)


