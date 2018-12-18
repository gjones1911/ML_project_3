from DataCleaner import *


def get_distance_knn(data):

    dic_list = list()

    main_dic = {}

    for obs in range(len(data)):
        tmp_dic = {}
        for r in range(len(data)):
            if r != obs:
                tmp_dic[r] = np.linalg.norm(data[obs] - data[r])
        main_dic[obs] = sorted(tmp_dic.items(), key=lambda kv: kv[1])

    return main_dic


#           test    train
def knn_vote(new_dp, train_d, k_val):

    dic_list = list()

    main_dic = {}

    # get the distances and sort in ascending order
    for obs in range(len(new_dp)):
        tmp_dic = {}
        for r in range(len(train_d)):
            tmp_dic[r] = np.linalg.norm(new_dp[obs] - train_d[r])
        main_dic[obs] = sorted(tmp_dic.items(), key=lambda kv: kv[1])

    vote_dic = {}
    # now vote
    # for each entry in the test set
    for ent in main_dic:
        # make a vote list for this entry in the test set
        vote_dic[ent] = list([0,0])
        # grab list of tuplse of distances from obs in train set
        tup_lst = main_dic[ent]
        cnt = 0
        # go through the tuple of distances starting from closest
        # and look at k closest and vote for 2 or 4 based on those
        for tup in tup_lst:
            obs = tup[0]
            if train_d[obs][len(train_d[obs])-1] == 2:
                vote_dic[ent][0] = vote_dic[ent][0] + 1
            else:
                vote_dic[ent][1] = vote_dic[ent][1] + 1
            cnt = cnt + 1
            if cnt == k_val:
                break

    return main_dic, vote_dic


#             test     train
def knn_vote2(dis_dic, tst_idx, trn_idx, d_set, kval):
    vote_dic = {}
    model = np.array(d_set)
    #print('tst_idx')
    #print(tst_idx)
    # go through test indices
    for idx in tst_idx:
    # for idx in trn_idx:
        # get the dis_dic for this observattion
        # and convert list of closest observations there to ints
        nn_l = list(map(int, dis_dic[idx]))
        #print('nn_l is')
        #print(nn_l)
        vote_dic[idx] = list([0, 0])
        cnt = 0

        # go through the test nearest neighbor indices
        # looking for k of them from the training indices
        for nn in nn_l:
            # go through the training indices
            for trn in trn_idx:
                # look for the next nn that is in training indices
                if nn == int(trn):
                    # if I found it in the training indices
                    # add one to found counter
                    cnt = cnt + 1
                    # use the found training index to look at data array
                    # and see what the nn predicts and add it's prediction to
                    # vote count vector
                    if d_set[int(trn)][len(d_set[int(trn)])-1] == 2:
                        vote_dic[idx][0] = vote_dic[idx][0] + 1
                    else:
                        vote_dic[idx][1] = vote_dic[idx][1] + 1
                    # once i found the nn break out of loop of training indices
                    break
                # if i didn't find the nn in the training set go to next nn
            # check to see if i found k of the nn's
            if cnt == kval:
                #print('cnt is {:d} at break'.format(cnt))
                break
    return vote_dic


def knn_vote3(dis_dic, tst_idx, model, kval):
    vote_dic = {}
    # go through test indices
    for idx in tst_idx:
        nn_l = list(map(int, dis_dic[idx]))
        vote_dic[idx] = list([0, 0])
        cnt = 0

        for nn in nn_l:
            if nn in model:
                cnt = cnt + 1
                if model[nn] == 2:
                    vote_dic[idx][0] = vote_dic[idx][0] + 1
                else:
                    vote_dic[idx][1] = vote_dic[idx][1] + 1
                if cnt == kval:
                    break

    return vote_dic


#             test     train
def create_model2(dis_dic, tst_idx, trn_idx, d_set, kval):
    vote_dic = {}
    model = np.array(d_set)

    ret_model = {}

    #print('tst_idx')
    #print(tst_idx)
    # go through test indices
    for idx in tst_idx:
    # for idx in trn_idx:
        # get the dis_dic for this observattion
        # and convert list of closest observations there to ints
        nn_l = list(map(int, dis_dic[idx]))
        #print('nn_l is')
        #print(nn_l)
        vote_dic[idx] = list([0, 0])
        cnt = 0

        # go through the test nearest neighbor indices
        # looking for k of them from the training indices
        for nn in nn_l:
            # go through the training indices
            for trn in trn_idx:
                # look for the next nn that is in training indices
                if nn == int(trn):
                    # if I found it in the training indices
                    # add one to found counter
                    cnt = cnt + 1
                    # use the found training index to look at data array
                    # and see what the nn predicts and add it's prediction to
                    # vote count vector
                    if d_set[int(trn)][len(d_set[int(trn)])-1] == 2:
                        vote_dic[idx][0] = vote_dic[idx][0] + 1
                    else:
                        vote_dic[idx][1] = vote_dic[idx][1] + 1
                    # once i found the nn break out of loop of training indices
                    break
                # if i didn't find the nn in the training set go to next nn
            # check to see if i found k of the nn's
            if cnt == kval:
                #print('cnt is {:d} at break'.format(cnt))
                break

    for idx in vote_dic:
        if vote_dic[idx][0] > vote_dic[idx][1]:
            ret_model[idx] = 2
            model[idx][-1] = 2
        else:
            ret_model[idx] = 4
            model[idx][-1] = 4

    return model, ret_model


# def knn_vote2(dis_dic, tst_idx, d_set, kval):
def create_model(dis_dic, trn_idx, d_set, kval):
    vote_dic = {}
    model = np.array(d_set)

    ret_model = {}

    # go through create indices
    for idx in trn_idx:
        # get the dis_dic for this observattion
        # and convert list of closest observations there to ints
        nn_l = list(map(int, dis_dic[idx]))
        vote_dic[idx] = list([0, 0])

        # go through the test nearest neighbor indices
        # looking for k of them from the training indices
        for nn in range(kval):
            if d_set[int(nn_l[nn])][-1] == 2:
                vote_dic[idx][0] = vote_dic[idx][0] + 1
            else:
                vote_dic[idx][1] = vote_dic[idx][1] + 1

    for idx in vote_dic:
        if vote_dic[idx][0] > vote_dic[idx][1]:
            ret_model[idx] = 2
            model[idx][-1] = 2
        else:
            ret_model[idx] = 4
            model[idx][-1] = 4

    return model, ret_model


def performance_calculator(cm, verbose=True, k=2):

    tp = float(cm[1][1])
    tn = float(cm[0][0])
    fp = float(cm[0][1])
    fn = float(cm[1][0])

    total = tp+tn+fp+fn

    accr = (tp + tn)/(total)

    sens = tp/(tp+fn)
    ppv = tp/(tp+fp)
    tnr = tn/(tn+fp)
    f1 = (2*ppv*sens)/(ppv+sens)

    #if verbose:
    #    display_confustion_matrix_tf(tn,tp,fn,fp)
    #    display_knn_performance(list([accr, sens, ppv, tnr, f1]), k)

    if verbose:
        return tp, fp, fn, tn, accr, sens, ppv, tnr, f1
    else:
        return accr, sens, ppv, tnr, f1


def make_confusion(data, tr_idx, v_dic):

    tp = 0     # if classed malignant and is
    tn = 0     # if classed benign and is
    fp = 0     # if classed malignant and isn't
    fn = 0     # if classed benign and isn't

    #    predicted
    #          b    m
    # true: ben[tn, fp]
    #       mal[fn, tp]

    for entry in v_dic:
        # print('observation ', entry)
        l = v_dic[entry]
        # if voted as benign
        if l[0] >= l[1]:
            # it it is benign
            if data[entry][len(data[entry])-1] == 2:
                tn = tn + 1
            else:
                fn = fn + 1
        # if i predict it as malignant
        elif l[0] < l[1]:
            # it it is malignant
            if data[entry][len(data[entry])-1] == 4:
                tp = tp + 1
            else:
                fp = fp + 1

    cm = [[tn, fp],
          [fn, tp]]

    # show_array(cm)

    # confusion_matrix(cm, verbose=False)

    accr, sens, ppv, tnr, f1 = performance_calculator(cm, verbose=False)

    return cm, accr, sens, ppv, tnr, f1


def knn_tester_data(data_np, k_val, num_runs):

    #data_np = load_data_file(file_name)

    accr_array = 0
    sens_array = 0
    ppv_array = 0
    tnr_array = 0
    f1_array = 0

    for run in range(num_runs):
        # main_dic = get_distance_knn(test_np)

        train_ary, test_ary, trn_idx, tst_idx = data_split(data_np[:])

        train_np = np.array(train_ary)
        test_np = np.array(test_ary)

        m_dic, vote_dic = knn_vote(test_np, train_np[:], k_val)

        cm, accr, sens, ppv, tnr, f1 = make_confusion(test_np, vote_dic)

        accr_array = accr_array + accr
        sens_array = sens_array + sens
        ppv_array = ppv_array + ppv
        tnr_array = tnr_array + tnr
        f1_array = f1_array + f1

    return accr_array/num_runs, sens_array/num_runs, ppv_array/num_runs, tnr_array/num_runs, f1_array/num_runs


def knn_tester(test_np, train_np, k_val, num_runs, cmret=False):

    #data_np = load_data_file(file_name)

    accr_array = 0
    sens_array = 0
    ppv_array = 0
    tnr_array = 0
    f1_array = 0

    for run in range(num_runs):
        # main_dic = get_distance_knn(test_np)

        #train_ary, test_ary = data_split(t_data[:])

        #train_np = np.array(train_ary)
        #test_np = np.array(test_ary)

        m_dic, vote_dic = knn_vote(test_np, train_np[:], k_val)

        cm, accr, sens, ppv, tnr, f1 = make_confusion(test_np, vote_dic)

        accr_array = accr_array + accr
        sens_array = sens_array + sens
        ppv_array = ppv_array + ppv
        tnr_array = tnr_array + tnr
        f1_array = f1_array + f1

    if not cmret:
        return accr_array/num_runs, sens_array/num_runs, ppv_array/num_runs, tnr_array/num_runs, f1_array/num_runs
    else:
        return accr_array/num_runs, sens_array/num_runs, ppv_array/num_runs, tnr_array/num_runs, f1_array/num_runs, cm


def knn(dis_dic, data, model, k_val, num_runs, test_idx, trn_idx, cmret=False):

    #data_np = load_data_file(file_name)

    accr_array = 0
    sens_array = 0
    ppv_array = 0
    tnr_array = 0
    f1_array = 0
    #print('num runs', num_runs)
    for run in range(num_runs):
        # main_dic = get_distance_knn(test_np)

        #train_ary, test_ary = data_split(t_data[:])

        #train_np = np.array(train_ary)
        #test_np = np.array(test_ary)

        # split_data(len(data))

        # get the votes dictionary
        vote_dic = knn_vote2(dis_dic, test_idx, trn_idx, model, k_val)
        #print('run {:d}'.format(run))
        #print(vote_dic)
        #cm, accr, sens, ppv, tnr, f1 = make_confusion(test_np, vote_dic)
        cm, accr, sens, ppv, tnr, f1 = make_confusion(data, test_idx, vote_dic)

        accr_array = accr_array + accr
        sens_array = sens_array + sens
        ppv_array = ppv_array + ppv
        tnr_array = tnr_array + tnr
        f1_array = f1_array + f1

    if not cmret:
        return accr_array/num_runs, sens_array/num_runs, ppv_array/num_runs, tnr_array/num_runs, f1_array/num_runs
    else:
        return accr_array/num_runs, sens_array/num_runs, ppv_array/num_runs, tnr_array/num_runs, f1_array/num_runs, cm


def knnb(dis_dic, data, model_dic, k_val, num_runs, test_idx, cmret=False):

    #data_np = load_data_file(file_name)

    accr_array = 0
    sens_array = 0
    ppv_array = 0
    tnr_array = 0
    f1_array = 0
    #print('num runs', num_runs)
    for run in range(num_runs):
        # main_dic = get_distance_knn(test_np)

        #train_ary, test_ary = data_split(t_data[:])

        #train_np = np.array(train_ary)
        #test_np = np.array(test_ary)

        # split_data(len(data))

        vote_dic = knn_vote3(dis_dic, test_idx, model_dic, k_val)

        cm, accr, sens, ppv, tnr, f1 = make_confusion(data, test_idx, vote_dic)

        accr_array = accr_array + accr
        sens_array = sens_array + sens
        ppv_array = ppv_array + ppv
        tnr_array = tnr_array + tnr
        f1_array = f1_array + f1

    if not cmret:
        return accr_array/num_runs, sens_array/num_runs, ppv_array/num_runs, tnr_array/num_runs, f1_array/num_runs
    else:
        return accr_array/num_runs, sens_array/num_runs, ppv_array/num_runs, tnr_array/num_runs, f1_array/num_runs, cm


def test_k_val(d_set, trn_set, k_val, n_r, verbose=False):
    accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg = knn_tester(d_set, trn_set, k_val, n_r)
    allavg = np.mean(list([accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg]), dtype=np.float64)

    if verbose:
        return accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, allavg
    else:
        return allavg


def test_k_val3(model_dic, d_set, val_idx, trn_idx, k_val, n_r, dis_dic, verbose=False):
    #accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg = knn_tester(d_set, trn_set, k_val, n_r)
    accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg = knnb(dis_dic, d_set, model_dic, k_val, n_r, val_idx, trn_idx)
    # knn(dis_dic, data, k_val, num_runs, test_idx, cmret=False):

    allavg = np.mean(list([accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg]), dtype=np.float64)

    if verbose:
        return accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, allavg
    else:
        return allavg


def test_k_val2(model, d_set, val_idx, trn_idx, k_val, n_r, dis_dic, verbose=False):
    #accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg = knn_tester(d_set, trn_set, k_val, n_r)
    accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg = knn(dis_dic, d_set, model, k_val, n_r, val_idx, trn_idx)
    # knn(dis_dic, data, k_val, num_runs, test_idx, cmret=False):

    allavg = np.mean(list([accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg]), dtype=np.float64)

    if verbose:
        return accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, allavg
    else:
        return allavg


def find_best_k_val(d_set, trn_set, k_val_set,n_r):

    avgr = 0
    best_k = 0
    ret_dic = {}
    for k in k_val_set:

        accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, c_avgr = test_k_val(d_set, trn_set, k, n_r, verbose=True)

        ret_dic[k] = list([accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg])

        if c_avgr > avgr:
            avgr = c_avgr
            best_k = k

    return ret_dic, avgr, best_k,


def find_best_k_valb(d_set, val_idx, trn_idx, k_val_set, n_r, dis_dic):
    ret_model = None
    avgr = 0
    best_k = 0
    ret_dic = {}
    #print('number of runs is ',n_r)
    print('Finding the best k.......')
    for k in k_val_set:

        # train with training set and get
        #accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, c_avgr = test_k_val2(d_set, trn_idx, trn_idx, k, n_r,
        #accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, c_avgr = test_k_val2(d_set, trn_idx, trn_idx, k, n_r,
        #                                                                   dis_dic, verbose=True)

        model, r_model = create_model(dis_dic, trn_idx, d_set, k)

        #accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, c_avgr = test_k_val2(d_set, val_idx, trn_idx, k, n_r,
        #                                                                   dis_dic, verbose=True)

        #accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, c_avgr = test_k_val2(model, d_set, val_idx, trn_idx, k, n_r,
        #                                                                   dis_dic, verbose=True)

        accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, cm = knn(dis_dic, d_set, model, k, n_r, val_idx, trn_idx,cmret=True )

        c_avgr = np.mean(list([accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg]), dtype=np.float64)

        # = test_k_val3(model_dic, val_idx, trn_idx, k_val, n_r, dis_dic, verbose=False):

        #print('k ', k)
        #print('accuracy', accr_avg)
        #print('sensitivity',sens_avg)
        #print('precision', ppv_avg)
        #print('tnr',tnr_avg)
        #print('fi ',f1_avg)
        #print('cumulative average', c_avgr)

        ret_dic[k] = list([accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, cm])

        if c_avgr > avgr:
            #print()
            #print('best k is now ', k)
            #print()
            avgr = c_avgr
            best_k = k
            ret_model = model

    return ret_dic, avgr, best_k, ret_model


def find_best_k_val2(d_set, tst_idx, trn_idx, k_val_set, n_r, dis_dic):

    avgr = 0
    best_k = 0
    ret_dic = {}
    for k in k_val_set:

        #accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, c_avgr = test_k_val(d_set, trn_set, k, n_r, verbose=True)
        accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg, c_avgr = test_k_val2(d_set, tst_idx, trn_idx, k, n_r, dis_dic,verbose=True)

        ret_dic[k] = list([accr_avg, sens_avg, ppv_avg, tnr_avg, f1_avg])

        if c_avgr > avgr:
            avgr = c_avgr
            best_k = k

    return ret_dic, avgr, best_k

'''
test_d = [[2,2],
          [4,4]]

data_np = load_data_file('breast-cancer-wisconsin.dt')

print(len(data_np))

#main_dic = get_distance_knn(test_np)

train_ary, test_ary = data_split(data_np[:])

train_np = np.array(train_ary)
test_np = np.array(test_ary)

m_dic, vote_dic_2 = knn_vote(test_np, train_np[:], 2)
m_dic, vote_dic_8 = knn_vote(test_np, train_np[:], 8)
m_dic, vote_dic_17 = knn_vote(test_np, train_np[:], 17)
m_dic, vote_dic_33 = knn_vote(test_np, train_np[:], 33)

'''
#for entry in vote_dic:
#    print('observation ', entry)
#    print(vote_dic[entry])


'''
print('---------------------------------------------')
print('k of 2')
con_m = make_confusion(test_np, vote_dic_2)
print('')
print('---------------------------------------------')
print('k of 8')
con_m = make_confusion(test_np, vote_dic_8)
print('')
print('---------------------------------------------')
print('k of 17')
con_m = make_confusion(test_np, vote_dic_17)
print('')
print('---------------------------------------------')
print('k of 33')
con_m = make_confusion(test_np, vote_dic_33)
print('')
print('---------------------------------------------')

    #print(main_dic[entry])
'''