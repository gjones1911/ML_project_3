from DataCleaner import *
from DisplayMethods import *
from performance_tests import *


# ####################################################################################################################
# ##################################     Class definitions     #######################################################
# ####################################################################################################################
class Question:
    """
        This represents a question to be asked by a node. It contains the column to check, tha value to check.
        It takes as arguments to its contructer the column(col) to check and the value to check(val)
        It has a method called answer that takes the row to check and returns true if the value at col in row is
        greater than or equal to the questions value. If not it returns false
    """
    def __init__(self, col, val, headers=None):
        self.col = col
        self.val = val
        self.headers = headers

    def set_headers(self, headers):
        self.headers = headers

    def answer(self, row):
        """
        if type(self.val) == 'string':
            if self.val.isnumeric():
                return row[self.col] >= self.val

            else:
                return row[self.col] == self.val
        else:
            return row[self.col] >= self.val
        """
        return int(row[self.col]) >= int(self.val)

    def __repr__(self):
        condition = '=='
        if (str(self.val)).isnumeric():
            condition = '>='
        if self.headers is not None:
            return 'Is item {:s} {:s} {:s}?'.format(self.headers[self.col], condition, str(self.val))
        else:
            return 'Is item in column {:s} {:s} {:s}?'.format(str(self.col), condition, str(self.val))


# ####################################################################################################################
# ##################################        Node definition         ##################################################
# ####################################################################################################################
# represents a node on a graph/tree for a decision tree
class Node:

    def __init__(self, data=None, headers=None, parent=None,
                 question=None, type=None, level=None, class_column=None, classes=None):
        self.data = data
        self.children = list()
        self.false_child = None
        self.true_child = None
        self.type = type
        self.question = question
        self.parent = parent
        self.headers = headers
        self.level = level
        self.class_column = class_column
        self.classes = classes
        if self.data is not None and self.class_column is not None:
            self.class_row = self.set_class_row()
        else:
            self.class_row = None

    # ############    Setters      #############################################################################
    def set_data(self, data):
        self.data = np.array(data)

    def set_headers(self, headers):
        self.headers = headers

    def set_question(self, col, val):
        if self.headers is None:
            self.question = Question(col, val)
        else:
            self.question = Question(col, val, headers=self.headers)

    def set_parent(self, parent):
        self.parent = parent

    def set_child(self, q):
        self.children.append(q)

    def set_type(self, n_type):
        self.type = n_type

    def set_false_child(self, new_child):
        self.false_child = new_child

    def set_true_child(self, new_child):
        self.true_child = new_child

    def set_level(self, level):
        self.level = level

    def set_class_column(self, col):
        if col > len(self.data[0]):
            print('Bad column must be less than or equal to {:d}'.format(len(self.data[0])-1))
        else:
            self.class_column = col

    def set_class_row(self):
        if self.data is not None and self.class_column is not None:
            self.class_row = self.data[:, self.class_column]
            return self.class_row
        else:
            return None

    def set_classes(self):
        if self.class_row is not None:
            self.classes = list(set([i for i in self.class_row]))
        return

    # ############    Getters      #############################################################################
    def get_data(self):
        return self.data

    def get_observation(self, row):
        return self.data[row]

    def get_col(self, col):
        if self.data is not None:
            return self.data[:, col]
        else:
            return None

    def get_element(self, r, c):
        return self.data[r, c]

    def get_child(self, tf):
        return self.children[tf]

    def get_children(self):
        return self.children

    def get_question(self):
        return self.question

    def get_parent(self):
        return self.parent

    def get_type(self):
        return self.type

    def get_false_child(self):
        return self.false_child

    def get_true_child(self):
        return self.true_child

    def get_level(self):
        return self.level

    def get_class_column(self):
        if self.class_column is not None:
            return self.class_column
        else:
            print('Class column not yet set')
            return None

    def get_num_observations(self):
        if self.data is not None:
            return len(self.data)
        else:
            print("Data not yet set")
            return None

    def get_num_attributes(self):
        if self.data is not None:
            return len(self.data[0])
        else:
            print("Data not yet set")
            return None

    # ###############################################   methods   ###################################################
    # this returns a prediction
    # it looks through this nodes data array at the class column predicts the class
    # that appears the most
    def get_prediction(self, verbose=False):
        if self.class_column is not None:
            row = self.data[:, self.class_column].tolist()
        else:
            row = self.data[:, len(self.data[0])-1].tolist()
        # get class options
        options_s = list(set([r for r in row]))
        if verbose:
            print('options_s')
            print(options_s)
        most = 0
        pred = -1
        # determine which class has highest number
        for o in options_s:
            m = row.count(o)
            if m > most:
                most = m
                pred = o
        if verbose:
            print('I predict : ', pred)
        return pred

# #####################################################################################################################
# #########################################     Methods        ########################################################


# returns the probability that a value is represented in a given row/list
def get_prob(row, val, N):
    return row.count(val)/N


# a generic form
def calculate_split_stats_gen(branches, vals, class_col):

    Nmb_array = list()
    prob_array = list()
    for i in range(len(branches)):
        ls = list()
        prob_array.append(ls)

    cnt = 0
    for branch in branches:
        class_row = branch[:, class_col].tolist()

        # number that took current branch
        Nmb_array.append(len(branch))

        for val in vals:
            # calculate the percentage of class 1 and class 2 in true branch
            prob_array[cnt].append(class_row.count(val) / len(branch))

    return Nmb_array, prob_array


# calculates and returns the number of rows in each branch,
# and the probability of each class in each node
# pmt1, pmt2 = probability of class1, class2 in true branch
# pmf1, pmf2 = probability of class1, class2 in false branch
def calculate_split_stats(branch1, branch2, val):
    row1 = branch1[:, len(branch1[0]) - 1].tolist()
    row2 = branch2[:, len(branch2[0]) - 1].tolist()

    # number that took true branch
    Nmt = len(branch1)

    # number that took false branch
    Nmf = len(branch2)

    # calculate the percentage of class 1 and class 2 in true branch
    pmt1 = row1.count(val) / Nmt
    pmt2 = 1 - pmt1

    # calculate the percentage of class 1 and class 2 in true branch
    pmf1 = row2.count(val) / Nmf
    pmf2 = 1 - pmf1
    return row1, row2, Nmt, Nmf, pmt1, pmt2, pmf1, pmf2


# used to display the stats of a split
# mainly used for debugging
def print_split_stats(Nm, Nmt, Nmf, pmt1, pmt2, pmf1, pmf2):
    print('Number in this node: ', Nm)
    print('Number that took true branch: ', Nmt)
    print('Probability of true branch: class 1 {:f}, class 2 {:f}', pmt1, pmt2)
    print('Number that took false branch', Nmf)
    print('Probability of false branch: class 1 {:f}, class 2 {:f}', pmf1, pmf2)
    return


# calculates the gini index of given row/list based
# assumes a two class classifier
def gini_index(branches, vals, Nm, class_col,verbose=False):
    Nmb_array, prob_array = calculate_split_stats_gen(branches, vals, class_col)

    imp = 0

    for Nmt, prb_row, in zip(Nmb_array, prob_array):
        pprod = 2
        for v in range(len(vals)):
            pprod *= prb_row[v]

        imp += (Nmt/Nm) * pprod

    return imp


# calculates the gini index of given row/list based
# assumes a two class classifier
def gini_index2(branch1, branch2, val, Nm, verbose=False):
    """calculates the gini index for a binary classifier split
    row1 = branch1[:, len(branch1[0])-1].tolist()
    row2 = branch2[:, len(branch2[0])-1].tolist()
    Nmt = len(branch1)
    # number that took balse branch
    Nmf = len(branch2)
    pmt1 = row1.count(val) / Nmt
    pmt2 = 1 - pmt1
    pmf1 = row2.count(val) / Nmf
    pmf2 = 1 - pmf1
    """
    row1, row2, Nmt, Nmf, pmt1, pmt2, pmf1, pmf2 = calculate_split_stats(branch1, branch2, val)

    true_val = (Nmt / Nm) * (2*pmt1 * pmt2)
    false_val = (Nmf / Nm) * (2*pmf1 * pmf2)

    if verbose:
        print_split_stats(Nm, Nmt, Nmf, pmt1, pmt2, pmf1, pmf2)
    return true_val+false_val


def entropy(row, val, Nm, verbose=False):
    p1 = get_prob(row, val, Nm)
    p2 = (1.0-p1)
    if verbose:
        print('p1, p2', p1, p2)
    return -p1 * np.log2(p1) - p2 * np.log2(p2)


# returns the entropy (impurity of the given split(branch1, branch2)
def entropy2(branch1, branch2, val, Nm, verbose=False):
    """Returns the entropy of the given split
    row1 = branch1[:, len(branch1[0]) - 1].tolist()
    row2 = branch2[:, len(branch2[0]) - 1].tolist()
    # number that took true branch
    Nmt = len(branch1)
    # number that took balse branch
    Nmf = len(branch2)
    pmt1 = row1.count(val)/Nmt
    pmt2 = 1 - pmt1
    pmf1 = row2.count(val)/Nmf
    pmf2 = 1 - pmf1
    """

    row1, row2, Nmt, Nmf, pmt1, pmt2, pmf1, pmf2 = calculate_split_stats(branch1, branch2, val)

    if pmt1 != 0:
        pmt1log = np.log2(pmt1)
    else:
        pmt1log = 0

    if pmt2 != 0:
        pmt2log = np.log2(pmt2)
    else :
        pmt2log = 0

    if pmf2 != 0:
        pmf2log = np.log2(pmf2)
    else:
        pmf2log = 0

    if pmf1 != 0:
        pmf1log = np.log2(pmf1)
    else :
        pmf1log = 0

    tprob = Nmt / Nm
    fprob = Nmf / Nm

    # true_val = tprob*(pmt1*np.log2(pmt1) + pmt2*np.log2(pmt2))
    true_val = tprob*(pmt1*pmt1log + pmt2*pmt2log)

    false_val = fprob*(pmf1*pmf1log + pmf2*pmf2log)

    Nm1 = len(row1)
    Nm2 = len(row2)

    if verbose:
        print(row1)
        print(row2)
    if verbose:
        print_split_stats(Nm, Nmt, Nmf, pmt1, pmt2, pmf1, pmf2)
        '''
        print('pb11, pb12', pmt1, pmt2)
        print('pb21, pb22', pmf1, pmf2)
        '''
    return -true_val - false_val


# returns the miss classification error of the given split set
def miss_class(row, val, Nm, verbose=False):
    p1 = get_prob(row, val, Nm)
    p2 = (1.0-p1)
    if verbose:
        print('p1, p2', p1, p2)
    return 1-max(p1,p2)


# returns the miss classification error of the given split set
def mis_class(branch1, branch2, val, Nm, verbose=False):
    """
    row1 = branch1[:, len(branch1[0]) - 1].tolist()
    row2 = branch2[:, len(branch2[0]) - 1].tolist()

    # number that took true branch
    Nmt = len(branch1)

    # number that took balse branch
    Nmf = len(branch2)

    pmt1 = row1.count(val) / Nmt
    pmt2 = 1 - pmt1

    pmf1 = row2.count(val) / Nmf
    pmf2 = 1 - pmf1

    Nm1 = len(row1)
    Nm2 = len(row2)
    """

    row1, row2, Nmt, Nmf, pmt1, pmt2, pmf1, pmf2 = calculate_split_stats(branch1, branch2, val)
    ret_val = (Nmt/Nm)*(1-max(pmt1,pmt2)) + (Nmf/Nm)*(1-max(pmf1,pmf2))

    if verbose:
        print(row1)
        print(row2)
    if verbose:
        print_split_stats(Nm,Nmt, Nmf, pmt1, pmt2, pmf1, pmf2)
    return ret_val


# returns the unique values of the given list as a set
def unique_vals_list(listl):
    return set([c for c in listl])


# determines the best split of the given array based on the impurity measure (class_e)
def best_split(array, clss, col_a, class_e='gini', verbose=False, imp_thrsh=None):
    if verbose:
        print('------------------------------------------------------------>getting best split')

    Nm = len(array)

    if Nm == 0:
        print('bad array')
        quit(-5)

    if verbose:
        print('given to split')
        print(array)

    var_set = list(range(1, 11))

    if verbose:
        print(var_set)
    best_imp = 100

    true_l = list()
    false_l = list()
    best_val = 0
    best_c = -1
    imp = -1
    # used to determine when a impurity threshold is hit
    thrsh_found = False

    for var in var_set:
        if thrsh_found:
            break
        if verbose:
            print('------------------------------------------------------------>looking at var :', var)

        # go through all the columns one by one
        for c in range(0, len(array[0])):
            if c == col_a:
                # ignore class column
                continue
            if thrsh_found:
                break
            if verbose:
                print('---------------------------->looking column :', c)
            t_l = list()
            f_l = list()
            # go through row by row comparing item in current column to current val
            for r in range(len(array)):
                if thrsh_found:
                    break
                num = array[r][c]
                if verbose:
                    print('checking against value {:d}'.format(num))
                if int(num) >= int(var):
                    if verbose:
                        print('it is greater')
                    # if the current entry is greater than or equal to val
                    # put current row in true branch
                    t_l.append(array[r].tolist())
                else:
                    # otherwise add row to false branch
                    f_l.append(array[r].tolist())

            tlnp = np.array(t_l)
            flnp = np.array(f_l)
            # if no split occured keep going
            if len(tlnp) == 0 or len(flnp) == 0:
                continue

            # perform indicated impurity calculation
            if class_e == 'gini':
                imp = gini_index2(tlnp, flnp, clss, Nm, verbose=verbose)
                if verbose:
                    print('imp: ', imp)
            elif class_e == 'entropy':
                imp = entropy2(tlnp, flnp, clss, Nm, verbose=verbose)
                if verbose:
                    print('imp: ', imp)
            elif class_e == 'miss class':
                imp = mis_class(tlnp, flnp, clss, Nm, verbose=verbose)
                if verbose:
                    print('imp: ', imp)
            if imp_thrsh is not None and 0 <= imp <= imp_thrsh:
                # print('found the threshold')
                thrsh_found = True
            # store the minimum impurity found and  the branchs
            # associated with it
            if imp < best_imp or thrsh_found:
                best_imp = imp
                best_c = c
                best_val = var
                true_l.clear()
                false_l.clear()
                true_l = list(t_l)
                false_l = list(f_l)

            if thrsh_found:
                break

    if verbose:
        print('best imp = {:f}, best c is {:d}, best val is {:d}'.format(best_imp, best_c, best_val))
    if verbose and imp_thrsh is not None and not thrsh_found:
        print('never found thresh hold')
    return best_imp, best_c, best_val, true_l, false_l


# creates a decision tree based on the given data
def make_tree(data, col_a=None, headers=None, type='root', error_type='gini', depth=0, depth_limit=None,
              verbose=False, imp_thrsh=None):

    if col_a is None:
        col_a = len(data[0])-1

    new_depth = None

    # the class values
    clsss = list(unique_vals_list(data[:, col_a]))

    # bi, bc, bv, t_l, f_l, f_pb, t_pb = best_split(node.get_data(),0, col_a)
    bi, bc, bv, t_l, f_l = best_split(data, clsss[0], col_a, class_e=error_type, imp_thrsh=imp_thrsh, verbose=verbose)

    if len(t_l) == 0 and len(f_l) == 0:
        print('should not see this')
        node = Node(data, type='leaf', headers=headers, level=depth, class_column=col_a, classes=clsss)
        node.set_question(bc, bv)
        return node, depth
    if verbose:
        print(t_l)
        print(f_l)

    # print('bi is ', bi)
    if bc != -1:
        node = Node(data, type=type, headers=headers, level=depth, class_column=col_a, classes=clsss)
        node.set_question(bc, bv)

    '''
    if bc == -1 or imp_thrsh is not None and imp_thrsh >= bi:
        if imp_thrsh is not None and bi <= imp_thrsh:
            if verbose:
                print('It hit the threshold')
        return Node(data, type='leaf', headers=headers, level=depth, class_column=col_a, classes=clsss), depth
    '''

    if len(f_l) > 0:

        fl = np.array(f_l)
        f_pb = get_prob(fl[:, len(fl[0])-1].tolist(), fl[0, len(fl[0])-1], len(fl[:, len(fl[0])-1].tolist()))
        if f_pb == 0 or f_pb == 1 or (depth_limit is not None and depth_limit == depth+1) or ( imp_thrsh is not None and bc <= imp_thrsh):
            if verbose:
                print('found a false leaf')
            rchild = Node(fl, type='leaf', headers=headers, level=depth+1, class_column=col_a)
            node.set_child(rchild)
            node.set_false_child(rchild)
            new_depthf = depth + 1
        else:
            falsechild, new_depthf = make_tree(fl, col_a, headers=headers, type='internal', error_type=error_type, depth=depth+1,
                                               verbose=verbose, imp_thrsh=imp_thrsh, depth_limit=depth_limit)
            if falsechild is not None:
                node.set_child(falsechild)
                node.set_false_child(falsechild)
            else:
                print('false child none')
    else:
        print('empty false list')

    if len(t_l) > 0:
        tl = np.array(t_l)
        t_pb = get_prob(tl[:, len(tl[0])-1].tolist(), tl[0, len(tl[0])-1], len(tl[:, len(tl[0])-1].tolist()))
        if t_pb == 0 or t_pb == 1 or (depth_limit is not None and depth_limit == depth+1) or ( imp_thrsh is not None and bc <= imp_thrsh):
            if verbose:
                print('found a true leaf')
            lchild = Node(tl, type='leaf', headers=headers, level=depth+1, class_column=col_a)
            node.set_child(lchild)
            node.set_true_child(lchild)
            new_deptht = depth+1
        else:
            truechild, new_deptht = make_tree(tl, col_a, headers=headers, type='internal', error_type=error_type, depth=depth+1,
                                  verbose=verbose, imp_thrsh=imp_thrsh, depth_limit=depth_limit)
            if truechild is not None:
                node.set_child(truechild)
                node.set_true_child(truechild)
            else:
                print('true child none')
    else:
        print('empty true list')

    new_depth = max(new_depthf, new_deptht)

    return node, new_depth


# creates predictions based on the given data
# using the root of a decision tree
def test_model(data, root, verbose=False):

    node = root

    predictions = list()
    actual = list()
    for observation in data:

        if verbose:
            print('row')
            print(observation.tolist())

        if verbose:
            print('actual: {:d}'.format(observation[len(observation)-1]))

        actual.append(observation[len(observation)-1])

        while node.get_type() != 'leaf':
            if verbose:
                print(node.question)
                print()
            if node.question.answer(observation):
                node = node.get_true_child()
            else:
                node = node.get_false_child()

        predictions.append(node.get_prediction())
        node=root

    return predictions, actual


# calculates the accuracy of predictions
def calculate_error(pred, actual):

    correct = 0
    for p,a in zip(pred,actual):

        if p == a:
            correct += 1

    return correct/len(pred)


# prints out the contents of a decision tree takes the root as an argument
def trace_tree(root):
    print('')
    if root.get_question() != None:
        print(root.get_question())
        print('I am a {:s}'.format(root.get_type()))
    else:
        print('I am a {:s}'.format(root.get_type()))

    print(root.data)
    print('')
    if root.get_type() != 'leaf':
        print('i have {:d} child'.format(len(root.get_children())))
        cnt = 1
        for q in root.get_children():
            print('--------------------------child {:d}--depth: {:d}---------------'.format(cnt, q.get_level()))
            trace_tree(q)
            cnt = cnt + 1
    else:
        print('i have {:d} child'.format(len(root.get_children())))
        root.get_prediction(verbose=True)


# finds the max value of a given dictionary
def max_dic_element(avg_err_dic, verbose=False, ret_key=True):
    best = 0
    b_depth = 0
    for depth in avg_err_dic:
        if avg_err_dic[depth] > best:
            best = avg_err_dic[depth]
            b_depth = depth

    if verbose:
        print('the best average at a depth of {:d} is {:f}'.format(b_depth, best))
    if ret_key:
        return b_depth, best
    else:
        return best


# performs series of testing runs(test_num) for my decision tree methods using a depth limit
def perform_tree_testing_depth_limit(data_np, depth_l, error_type='entropy', class_column=None, verbose=False,
                                     headers=[]):

    print('\nerror type', error_type)
    print('Creating and testing model decision trees using different depth limits...')

    trn_cm = {}
    test_cm = {}
    trn_per = {}
    test_per = {}

    depth_o_bst = {}

    psum_dic = {}
    psum_dic[1] = 1
    best_depth = 0
    best_err = {}
    best_mean_acc = 0
    tr_dic = {}
    train_dic = {}
    ts_dic = {}
    val_dic = {}
    model_dic = {2: None, 8: None, 17: None, 33: None}

    # test_array = generate_testing_sets(len(data_np), test_num, p_train=.80, p_test=.15, p_val=.05, verbose=verbose)
    test_array = generate_testing_sets(len(data_np), 1, p_train=.80, p_test=.10, p_val=.10, verbose=verbose)
    tr, ts, vs = test_array[0][0], test_array[1][0], test_array[2][0]
    train_set = np.array(get_cross_array(data_np, tr))
    test_set = np.array(get_cross_array(data_np, ts))
    if len(vs) != 0:
        val_set = np.array(get_cross_array(data_np, vs))

    for depth_limit in depth_l:
        if verbose:
            print(
                '----------------------------------------------------------------->>  depth limit {:d}'.format(depth_limit))

        root, tree_depth = make_tree(train_set, class_column,
                         headers=headers, error_type=error_type, depth_limit=depth_limit, imp_thrsh=None, depth=0)

        depth_o_bst[depth_limit] = tree_depth

        print('tree depth is : ', tree_depth)

        predictions, actual = test_model(train_set, root)
        cm, performance_array = confusion_matrix(predictions, actual)
        trn_cm[depth_limit] = cm
        trn_per[depth_limit] = performance_array

        predictions, actual = test_model(test_set, root)
        cm, performance_array = confusion_matrix(predictions, actual)
        c_mean = np.mean(performance_array, dtype=np.float64)
        test_cm[depth_limit] = cm
        test_per[depth_limit] = performance_array

        # if err > best_err[depth_limit]:
        if c_mean > best_mean_acc:
            best_mean_acc = c_mean
            best_depth = depth_limit
            best_err[depth_limit] = c_mean
            train_dic[depth_limit] = train_set
            tr_dic[depth_limit] = list(map(int, tr))
            ts_dic[depth_limit] = test_set
            val_dic[depth_limit] = val_set
            model_dic[depth_limit] = root

    performance_array = [[trn_cm, trn_per],
                         [test_cm, test_per]]
    tr_ts_vl_array = [train_dic,    # 0
                      tr_dic,       # 1
                      ts_dic,       # 2
                      val_dic]      # 3

    return best_mean_acc, depth_o_bst, tr_ts_vl_array, model_dic, best_depth, performance_array


# performs series of testing runs(test_num) for my decision tree methods using a impurity threshold
def perform_tree_testing_imp_limit(data_np, imp_thl, error_type='entropy', class_column=None, verbose=False,
                                   headers=[]):
    print('\nerror type', error_type)
    print('Creating and testing model decision trees using different impurity thresholds...')

    stat_labels = ['acc', 'sensitivity', 'precision', 'TNR', 'F1']

    trn_cm = {}
    test_cm = {}
    trn_per = {}
    test_per = {}

    best_imp = 0

    best_err = {}
    depth_o_bst = {}
    best_mean_acc = 0
    tr_dic = {}
    train_dic = {}
    ts_dic = {}
    val_dic = {}
    model_dic = {}

    last_depth = 0

    test_array = generate_testing_sets(len(data_np), 1, p_train=.80, p_test=.10, p_val=.10, verbose=verbose)
    tr, ts, vs = test_array[0][0], test_array[1][0], test_array[2][0]
    train_set = np.array(get_cross_array(data_np, tr))
    test_set = np.array(get_cross_array(data_np, ts))
    val_set = np.array(get_cross_array(data_np, vs))

    for imp_thrsh in imp_thl:
        #if verbose:
        if True:
            print(
                '--------------------------------------------------------->>impurity threshold {:f}'.format(imp_thrsh))
        for t in range(1):
            '''
            tr, ts, vs = test_array[0][t], test_array[1][t], test_array[2][t]
            train_set = np.array(get_cross_array(data_np, tr))
            test_set = np.array(get_cross_array(data_np, ts))            
            if len(vs) != 0:
                val_set = np.array(get_cross_array(data_np, vs))
            '''

            root, tree_depth = make_tree(train_set, class_column,
                                         headers=headers, error_type=error_type, depth_limit=None, imp_thrsh=imp_thrsh)

            depth_o_bst[imp_thrsh] = tree_depth

            if tree_depth != last_depth:
                print('tree depth is : ', tree_depth)

            if tree_depth == 0:
                continue

            predictions, actual = test_model(train_set, root)
            cmtr, performance_arraytr = confusion_matrix(predictions, actual)
            if last_depth != tree_depth:
                trn_cm[imp_thrsh] = cmtr
                trn_per[imp_thrsh] = performance_arraytr

            # test on validation set and store results for this limit
            predictions, actual = test_model(test_set, root)
            cmts, performance_arrayts = confusion_matrix(predictions, actual)
            c_mean = np.mean(performance_arrayts, dtype=np.float64)
            if last_depth != tree_depth:
                test_cm[imp_thrsh] = cmts
                test_per[imp_thrsh] = performance_arrayts
                last_depth = tree_depth

            if verbose:
                print('average at {:f} for impurity threshold {:f}'.format(c_mean, imp_thrsh))
                print('performance array')
                for idx in range(len(performance_arrayts)):
                    print(stat_labels[idx], ': ', performance_arrayts[idx])

            if c_mean > best_mean_acc:
                print('best imp is now: ', imp_thrsh)
                trn_cm[imp_thrsh] = cmtr
                trn_per[imp_thrsh] = performance_arraytr
                test_cm[imp_thrsh] = cmts
                test_per[imp_thrsh] = performance_arrayts

                best_mean_acc = c_mean
                best_imp = imp_thrsh
                train_dic[imp_thrsh] = root
                tr_dic[imp_thrsh] = list(map(int, tr))
                ts_dic[imp_thrsh] = test_set
                val_dic[imp_thrsh] = val_set
                model_dic[imp_thrsh] = root

    performance_array = [[trn_cm, trn_per],
                         [test_cm, test_per]]
    tr_ts_vl_array = [train_dic,  # 0
                      tr_dic,     # 1
                      ts_dic,     # 2
                      val_dic]    # 3

    return best_mean_acc, depth_o_bst, tr_ts_vl_array, model_dic, best_imp, performance_array


# can be used to display results of testing but also returns the best avg accuracy
# and the depth level where this cccured as well as the highest accuracy and the depth
#  where this occured
def display_results_mulit_test(best_err, psum_dic, depths, imp=False, verbose=False):

    if verbose:
        for depth in depths:
            if not imp:
                print('the best accuracy for depth limit {:d}  is {:f}'.format(depth, best_err[depth]))
            else:
                print('the best accuracy for impurity limit {:f}  is {:f}'.format(depth, best_err[depth]))
    if verbose:
        print()
        print()
    if verbose:
        for depth in depths:
            if not imp:
                print('the best avg accuracy for depth limit {:d}  is {:f}'.format(depth, psum_dic[depth]))
            else:
                print('the best avg accuracy for impurity limit {:f}  is {:f}'.format(depth, psum_dic[depth]))

    # write the best depth limit and the tree model that got it to a file
    b_depth, b_avg = max_dic_element(psum_dic, verbose=False)
    b_single_depth, b_single_acc = max_dic_element(best_err, verbose=False)
    if verbose:
        if not imp:
            print('the best avg accuracy of {:f} is achieved by {:d}'.format(b_avg, b_depth))
            print('the best single accuracy of {:f} is achieved by {:d}'.format(b_single_acc, b_single_depth))
        else:
            print('the best avg accuracy of {:f} is achieved by {:f}'.format(b_avg, b_depth))
            print('the best single accuracy of {:f} is achieved by {:f}'.format(b_single_acc, b_single_depth))

    return b_depth, b_avg, b_single_depth, b_single_acc


def display_result(best_err, avg_er):
    print('the best error is {:f}'.format(best_err))
    print()
    print('the best avg error is {:f}'.format(avg_er))
    return


# used to run training and testing of a decision tree and tests tree on validation set and returns
# confusion matrix and basic stats of model
def create_test_d_tree(data_np, er_idx, class_column, limit_type, depth_limit=None, imp_thrsh=None, headers=None):

    error_type = ['entropy', 'gini', 'miss class']

    if headers is None:
        headers = list()
        for i in range(len(data_np[0])):
            headers.append(str(i))

    if limit_type == 1 and depth_limit is None:
        depth_limit = list(range(2,11))
    elif limit_type == 2 and imp_thrsh is None:
        imp_thrsh = list([.2, .19, .18, .17, .16, .15, .14, .13, .12, .11, .10, .09, .08, .07, .06, .05, .04,
                          .03, .02, .01, .002, .0019, .0018, .0017, .0016, .0015, .0014, .0013, .0012, .0011, .0010,
                          .0009, .0008, .0007, .0006, .0005, .0004, .0003, .0002, .0001])


    # this uses a depth limit
    if limit_type == 1 or imp_thrsh is None:

        # b_mu_ac, b_acc, tr_dic, tr_dic, ts_dic, val_dic, mod_dic, b_depth = perform_tree_testing_depth_limit(data_np,
        # b_mu_ac, b_acc, rsv_a, mod_dic, b_depth, p_a = perform_tree_testing_depth_limit(data_np,
        b_mu_ac, b_tree_d, rsv_a, mod_dic, b_depth, p_a = perform_tree_testing_depth_limit(data_np,
                                                                                           depth_limit,
                                                                                           error_type[er_idx],
                                                                                           class_column,
                                                                                           verbose=False,
                                                                                           headers=headers)

        trn_cm = p_a[0][0]
        trn_perf = p_a[0][1]
        val_cm = p_a[1][0]
        val_perf = p_a[1][1]

        training_set = rsv_a[0]
        training_idx = rsv_a[1]
        validation_set = rsv_a[2]
        test_set = rsv_a[3]


        #b_avg_depth, b_avg = max_dic_element(b_acc, verbose=False)

        mod = mod_dic[b_depth]

        print('best depth ', b_depth)
        print('len of rsv ', len(rsv_a))

        mod = mod_dic[b_depth]

        print('--------------------------------------------------------')
        print('--------------------------------------------------------')
        print('                      Training set:\n')
        for d in range(2,11):
            print('Depth limit: ',d)
            print('Depth of tree: ', b_tree_d[d])
            dis_cmplx_perorm(trn_cm[d], trn_perf[d])
        print('--------------------------------------------------------')
        print('--------------------------------------------------------')
        print('                      Validation set:\n')
        for d in range(2, 11):
            print('Depth limit: ',d)
            print('Depth of tree: ', b_tree_d[d])
            dis_cmplx_perorm(val_cm[d], val_perf[d])

        print('--------------------------------------------------------')
        print('--------------------------------------------------------')
        print('                      Testing set:\n')
        print('Best tested depth limit: ', b_depth)
        print('Depth of tree: ', b_tree_d[b_depth])

        predictions, actual = test_model(test_set[b_depth], mod)
        cm, performance_array = confusion_matrix(predictions, actual)
        dis_cmplx_perorm(cm, performance_array)
        accuracy = performance_array[0]
        '''
        print('look here ')
        for p in performance_array:
            print(np.around(p, 2))
        '''
        return accuracy, mod

    elif limit_type == 2:

        b_mu_ac, b_tree_d, rsv_a, mod_dic, b_imp_tsh, p_a = perform_tree_testing_imp_limit(data_np,
                                                                                        imp_thrsh,
                                                                                        error_type[er_idx],
                                                                                        class_column,
                                                                                        verbose=False,
                                                                                        headers=headers)

        trn_cm = p_a[0][0]
        trn_perf = p_a[0][1]

        val_cm = p_a[1][0]
        val_perf = p_a[1][1]

        training_set = rsv_a[0]
        training_idx = rsv_a[1]
        validation_set = rsv_a[2]
        test_set = rsv_a[3]


        print('best impurity threshold ', b_imp_tsh)
        print('len of rsv ', len(rsv_a))

        print('it ', len(test_set))

        mod = mod_dic[b_imp_tsh]

        print('--------------------------------------------------------')
        print('--------------------------------------------------------')
        print('                      Training set:\n')
        #for i in imp_thrsh:
        for i in trn_cm:
            print('Impurity threshold: ', i)
            print('Depth of tree: ', b_tree_d[i])
            dis_cmplx_perorm(trn_cm[i], trn_perf[i])
        print('--------------------------------------------------------')
        print('--------------------------------------------------------')
        print('                      Validation set:\n')
        #for i in imp_thrsh:
        for i in val_cm:
            print('Impurity threshold: ', i)
            print('Depth of tree: ', b_tree_d[i])
            dis_cmplx_perorm(val_cm[i], val_perf[i])

        print('--------------------------------------------------------')
        print('--------------------------------------------------------')
        print('                      Testing set:\n')
        print('Best tested impurity threshold: ', b_imp_tsh)
        print('Depth of tree: ', b_tree_d[b_imp_tsh])

        predictions, actual = test_model(test_set[b_imp_tsh], mod)
        cm, performance_array = confusion_matrix(predictions, actual)
        dis_cmplx_perorm(cm, performance_array)

        print('--------------------------------------------------------')
        print('--------------------------------------------------------')
        accuracy = performance_array[0]
        return accuracy, mod


# ###############################################################################################################
# ###############################################################################################################
# ###############################################################################################################
# ###############################################################################################################
# ###############################################################################################################
# ###############################################################################################################
# ###############################################################################################################
# ###############################################################################################################
