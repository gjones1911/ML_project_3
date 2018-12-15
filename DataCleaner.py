import numpy as np
from DisplayMethods import *


# #####################################################################################################################
# #####################          Data saving and loading methods         ##############################################

def create_data_file_from_file(file_name, created_file='', ident='?'):
    if created_file == '':
        created_file = file_name[:file_name.index('.')] + '.dt'

    f = open(created_file, 'w')

    # len(lines), len(line_vec), avg_impute(d_np, avg_a, bd_obs), id_list, bd_obs

    num_obs, num_dim,  data_array, id, bd_obs = make_data_array(file_name, ident=ident)

    for r in range(len(data_array)):
        for c in range(len(data_array[0])):
            if c < len(data_array[0]) - 1:
                f.write(str(data_array[r][c]) + ',')
            else:
                f.write(str(data_array[r][c]) + '\n')

    dis_dic = get_knn(data_array)

    nn_file = created_file[:created_file.index('.')] + '.nn'

    write_dis_file2(nn_file, dis_dic)

    f.close()
    return


def load_data_file(file_name):

    f = open(file_name, 'r')

    lines = f.readlines()

    num_vec = list()

    for line in lines:
        line_vec = list(map(float, line.strip('\n').split(',')))
        num_vec.append(line_vec)
    f.close()
    return np.array(num_vec, dtype=np.float64)


def write_dis_file(file_name, dis_dic):

    f = open(file_name, 'w')

    for entry in dis_dic:

        obser_nm = entry
        f.write(obser_nm)
        f.write(':')

        tup_l = dis_dic[obser_nm]
        for i in range(len(tup_l)):
            tup = tup_l[i]
            f.write(tup[0])
            f.write(',')
            f.write(tup[1])
        f.write('\n')

    f.close()
    return


def write_dis_file2(file_name, dis_dic):

    f = open(file_name, 'w')

    for entry in dis_dic:

        obser_nm = entry
        f.write(str(obser_nm))
        f.write(':')

        l = dis_dic[obser_nm]
        cnt = 0
        for nn in l:
            f.write(str(nn))
            if cnt < len(l)-1:
                f.write(',')
            cnt = cnt + 1
        f.write('\n')

    f.close()
    return


def write_d_tree_file(file_name, avg_err, depth_limit, train_idx, imp_thrsh=None, verbose=False,
                      data_file=None):

    f = open(file_name, 'w')

    if data_file is not None:
        f.write('df'+'\n')
        f.write(data_file + '\n')

    # store average error attained by this tree builder
    f.write(str(avg_err)+ '\n')

    # store the depth limit used to get this
    f.write(str(depth_limit)+ '\n')

    last = len(train_idx)
    cnt = 0
    for idx in train_idx:
        if cnt < last-1:
            f.write(str(idx) + ',')
        else:
            f.write(str(idx))
        cnt += 1
        if cnt == last:
            f.write('\n')
    return


def read_d_tree(file_name):
    f = open(file_name, 'r')

    lines = f.readlines()
    avg_err = 0
    depth_limit = 0
    tr_idx = list()

    for idx in range(len(lines)):
        if idx == 0:
            # get the average error
            avg_err = float(lines[idx].strip('\n'))
        elif idx == 1:
            depth_limit = int(lines[idx].strip('\n'))
        else:
            tr_idx = lines[idx].strip('\n').strip('').split(',')
            #for val in tr:
            #    tr_idx.append(int(val))

    #print(tr_idx)
    tr_idx = list(map(int, tr_idx))

    return avg_err, depth_limit, tr_idx


def read_dis_file(file_name):
    f = open(file_name, 'r')
    ret_dic = {}
    for line in f.readlines():
        key, dis_dic = process_dis_line(line)
        ret_dic[key] = dis_dic
    f.close()
    return ret_dic


def read_dis_file2(file_name):
    f = open(file_name, 'r')
    ret_dic = {}
    for line in f.readlines():
        key, nn_l = process_dis_line2(line)
        ret_dic[key] = nn_l
    f.close()
    return ret_dic


def process_dis_line(line, key_del=':', obs_dil=','):
    if obs_dil == ',':

        # get where in line the key/row ends
        key_p = line.index(key_del)

        # get key/row from line
        key_val = int(line[0: key_p])

        # grab observations rows in ascending order of distance
        #
        dis_array_str = line[key_p:].strip('\n').split(',')

        # turn it into a list of floats
        dis_array = list(map(float, dis_array_str))

        dis_dic = {}

        nn_dic = {}
        nn_li = list()

        for i in range(0, len(dis_array), 2):
            # observation number = key
            # distance from = val
            dis_dic[dis_array[i]] = dis_array[i + 1]
            nn_li.append(dis_array[i])

        return key_val, dis_dic, nn_li


def process_dis_line2(line, key_del=':', obs_dil=''):
    if obs_dil == '':

        key_p = line.index(key_del)

        key_val = int(line[0: key_p])

        dis_array_str = line[key_p+1:].strip('\n').split(',')

        #dis_array = list(map(float, dis_array_str))

        nn_li = list(map(float, dis_array_str))

        return key_val, nn_li


######################################################################################################################
######################################################################################################################
def avg_impute(d_np, avg_a, bd_a):

    for row in bd_a:
        for col in bd_a[row]:
            d_np[row][col] = np.around(avg_a[col], 0)

    return d_np


def get_missing_data_points(line, ident='?'):
    '''
    This will take a row/line and find all columns were there is missing data
    and return a list of these, as well as the line with a -99 string replacing
    the bad data point
    :param line: the line to be searched
    :param ident: used to signify missing data point in column
    :return:
    '''
    ret_list = list()

    for col in range(len(line)):
        if line[col] == ident:
            ret_list.append(col)
            line[col] = '-99'

    return ret_list, line


def data_split(data, p_train=.70, p_test=.30, p_val=0):

    d = data.tolist()

    train_l = list()
    test_l = list()
    val_l = list()

    trn_idx = list()
    tst_idx = list()
    val_idx = list()
    r_c = np.random.choice(range(len(data)), len(data), replace=False)

    train = int(np.around(len(data) * p_train, 0))
    test = 0
    val = 0
    if p_val != 0:
        test = int(np.around(len(data) * p_test, 0,))
        val = len(data) - train - test
    else:
        test = len(data) - train

    for i in range(0, train):
        train_l.append(d[r_c[i]])
        trn_idx.append(r_c[i])

    for i in range(train, train+test):
        test_l.append(d[r_c[i]])
        tst_idx.append(r_c[i])

    for i in range(train+test, len(data)):
        val_l.append(d[r_c[i]])
        val_idx.append(r_c[i])



   # print(len(r_c))
    print('train: ', train)
    print('test: ', test)
    print('val: ', val)

    if val == 0:

        tr_np = np.array(train_l, dtype=np.float64)
        ts_np = np.array(test_l, dtype=np.float64)

        return tr_np, ts_np, trn_idx, tst_idx
    else:
        tr_np = np.array(train_l, dtype=np.float64)
        ts_np = np.array(test_l, dtype=np.float64)
        val_np = np.array(val_l, dtype=np.float64)
        return tr_np, ts_np, val_np, trn_idx, tst_idx, val_idx


def generate_tr_ts_vl(data_len, p_train=.80, p_test=.10, p_val=.1, verbose=False):
    return split_data(data_len, p_train=p_train, p_test=p_test, p_val=p_val, verbose=verbose)


def generate_testing_sets(data_len, num_rn, p_train=.80, p_test=.10, p_val=.1, verbose=False):

    test_array = [list(), list(), list()]
    for r in range(num_rn):
        tr, ts, vs = generate_tr_ts_vl(data_len, p_train, p_test, p_val, verbose)
        test_array[0].append(tr)
        test_array[1].append(ts)
        test_array[2].append(vs)
    return test_array

def split_data(data_size, p_train=.70, p_test=.30, p_val=0, verbose=False):

    trn_idx = list()
    tst_idx = list()
    val_idx = list()

    r_c = np.random.choice(range(data_size), data_size, replace=False)

    train = int(np.around(data_size * p_train, 0))
    test = 0
    val = 0
    if p_val != 0:
        test = int(np.around(data_size * p_test, 0,))
        val = data_size - train - test
    else:
        test = data_size - train

    if verbose:
        print('train set size: ', train)
        print('test set size: ', test)
        print('val set size: ', val)


    for i in range(0, train):
        trn_idx.append(r_c[i])

    for i in range(train, train+test):
        tst_idx.append(r_c[i])

    for i in range(train+test, data_size):
        val_idx.append(r_c[i])



    if val == 0:
        return trn_idx, tst_idx
    else:
        return trn_idx, tst_idx, val_idx


def get_unique_elements(dlist):
    return set([element for element in dlist])


def calc_probs(data, class_col, verbose=False):

    classes = get_unique_elements(data[:, class_col])
    print()
    c_l = list(classes)
    prb = {}
    for item in c_l:
        prb[item] = (data[:, class_col].tolist().count(item)/len(data))

    if verbose:
        for entry in prb:
            print('entry {:d} prob {:f}'.format(int(entry), prb[entry]))
    return prb, classes


def split_data_classifier(data, col_size, p_train=.70, p_test=.30, p_val=0,verbose=False):

    data_size = len(data)

    value_list = list(range(data_size))

    trn_idx = list()
    tst_idx = list()
    val_idx = list()

    prb, cls = calc_probs(data, col_size -1)

    classes = list(cls)

    #r_c = np.random.choice(range(data_size), data_size, replace=False)

    train = int(np.around(data_size * p_train, 0))

    train_c1 = int(np.around(train * prb[classes[0]], 0))
    train_c2 = train - train_c1

    #print('train is ', train)
    #print('train c1 is ', train_c1)
    #print('train c2 is ', train_c2)
    #print('train total is ', train_c2+train_c1)

    test = 0
    val = 0
    if p_val != 0:
        test = int(np.around(data_size * p_test, 0,))
        val = data_size - train - test
    else:
        test = data_size - train
        test_c1 = int(np.around(test * prb[classes[0]], 0))
        test_c2 = test - test_c1

    if verbose:
        print('test is ', test)
        print('test c1 is ', test_c1)
        print('test c2 is ', test_c2)
        print('test total is ', test_c2 + test_c1)

    #quit(-4)
    cnt = 0
    tr_c1_cnt = 0
    tr_c2_cnt = 0
    while cnt < train:
        rc = np.random.choice(value_list, 1, replace=False)
        if data[rc, col_size-1] == classes[0] and tr_c1_cnt < train_c1:
            cnt = cnt+1
            tr_c1_cnt = tr_c1_cnt + 1
            trn_idx.append(rc[0])
            del value_list[value_list.index(rc[0])]
        elif data[rc, col_size-1] == classes[1] and tr_c2_cnt < train_c2:
            cnt = cnt+1
            tr_c2_cnt = tr_c2_cnt + 1
            trn_idx.append(rc[0])
            del value_list[value_list.index(rc[0])]
        #print('cnt is : ',cnt)



    cnt = 0
    ts_c1_cnt = 0
    ts_c2_cnt = 0
    while cnt < test:
        rc = np.random.choice(value_list, 1, replace=False)
        rc = rc.tolist()
        if data[rc, col_size-1] == classes[0] and ts_c1_cnt < test_c1:
            cnt = cnt+1
            ts_c1_cnt = ts_c1_cnt + 1
            tst_idx.append(rc[0])
            del value_list[value_list.index(rc[0])]
        elif data[rc, col_size-1] == classes[1] and ts_c2_cnt < test_c2:
            cnt = cnt+1
            ts_c2_cnt = ts_c2_cnt + 1
            tst_idx.append(rc[0])
            del value_list[value_list.index(rc[0])]
        #print('cnt ts is : ',cnt)

    if verbose:
        print('train: ', train)
        print('test: ', test)
        print('val: ', val)

    if val == 0:
        return trn_idx, tst_idx
    else:
        return trn_idx, tst_idx, val_idx


def get_cross_array(data, indices):

    dl = data.tolist()

    ret_l = list()

    for idx in indices:
        ret_l.append(dl[idx])

    return ret_l


def make_data_array(file_name, ident='?'):

    ret_l = list()

    id_list = list()

    bd_obs = {}

    ok_obs = list()

    f = open(file_name, 'r')

    lines = f.readlines()

    cnt = 0
    for line in lines:

        line_vec = line.strip('\n').split(',')

        # grab the id of observation i
        id_list.append(line_vec[0])

        # remove the id
        del line_vec[0]
        # ret_l.append(line_vec)
        # if the current line has missing data
        # store what row it occurs
        # if line_vec.index('?', -99) != -99:
        if line.find(ident) != -1:
            c_list, line_vec = get_missing_data_points(line_vec, ident=ident)
            bd_obs[cnt] = c_list
        # otherwise store the ok observation for average imputation
        else:
            ok_obs.append(list(map(float, line_vec)))

        ret_l.append(list(map(float, line_vec)))

        # line_vec = list(map(float, line.split(',')))
        cnt = cnt + 1
    # make an array of the averages of the of the columns of rows
    avg_a = np.array(ok_obs, dtype=np.float64).mean(axis=0, dtype=np.float64)

    d_np = np.array(ret_l, dtype=np.float64)

    print('The bad rows are:')
    print(bd_obs)

    return len(lines), len(line_vec), avg_impute(d_np, avg_a, bd_obs), id_list, bd_obs


def get_knn(data):
    main_dic = {}

    for obs in range(len(data)):
        tmp_dic = {}
        for r in range(len(data)):
            if r != obs:
                tmp_dic[r] = np.linalg.norm(data[obs] - data[r])
        main_dic[obs] = sorted(tmp_dic.items(), key=lambda kv: kv[1])

    nn_dic = {}

    for obs in main_dic:
        tup_l = main_dic[obs]
        nn_l = list()
        for tup in tup_l:
            nn_l.append(tup[0])
        nn_dic[obs] = nn_l

    return nn_dic


def make_distance_dic(data):

    dic_list = list()

    main_dic = {}

    for obs in range(len(data)):
        tmp_dic = {}
        for r in range(len(data)):
            if r != obs:
                tmp_dic[r] = np.linalg.norm(data[obs] - data[r])
        main_dic[obs] = sorted(tmp_dic.items(), key=lambda kv: kv[1])

    return main_dic
# ####################################################################################################################
# ########################        cleaning methods        ############################################################
# ####################################################################################################################


def multi_strip(strg, to_strp):
    '''

    :param strg: string to be striped
    :param to_strp: what to strip the string of
    :return:
    '''
    cnt = strg.count(to_strp)
    for i in range(cnt):
        strg = strg.strip(to_strp)

    return strg






'''
num_obs, num_dim, data_array, ids, bd_obs= make_data_array('breast-cancer-wisconsin.data')
print('num_obs ', num_obs)
print('num_dim ', num_dim)
#print(data_array)
#show_array_selection(data_array.tolist(), bd_obs)


train_set, test_set, val_set = data_split(data_array, p_train=.70, p_test=.10, p_val=.20)
train_set2, test_set2 = data_split(data_array, p_train=.70, p_test=.10)
#data_split()

print('length of train set', len(train_set))
print('length of test set',len(test_set))
print('length of val set',len(val_set))


print('length of train set2', len(train_set2))
print('length of test set2',len(test_set2))
'''

'''
create_data_file_from_file('breast-cancer-wisconsin.data')

imp_data = load_data_file('breast-cancer-wisconsin.dt')

show_array(imp_data.tolist())
'''