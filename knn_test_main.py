from knn_estimator import *

data_np = load_data_file('breast-cancer-wisconsin.dt')
dis_dic = read_dis_file2('breast-cancer-wisconsin.nn')

k_vals = list([2, 8, 17, 33])

trn_idx, tst_idx, val_idx = split_data(len(data_np), p_train=.80, p_test=.10, p_val=.10)
print('total is {:d}'.format(len(trn_idx)+len(tst_idx) + len(val_idx)))
print('Train data size ', len(trn_idx))
print('val data size ', len(val_idx))
print('test data size ', len(tst_idx))

ret_dic, avgr, best_k, model = find_best_k_valb(data_np, val_idx, trn_idx, list([2, 8, 17, 33]), 1, dis_dic)

print('------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------')
print('----------------------------------------->THE BEST k IS: ', best_k)
print('------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------')
print('                                           For Training set:')
for kv in list([2, 8, 17, 33]):
    print('for k = {:d}'.format(kv))
    ac, sen, pv,  tnr, f1, cm = knn(dis_dic, data_np, model, kv, 1, trn_idx, trn_idx, cmret=True)
    dis_cmplx_perorm(cm, list([ac, sen, pv, tnr, f1]))
print('------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------')
print('                                            For Validation set:')
for kv in list([2, 8, 17, 33]):
    print('for k = {:d}'.format(kv))
    dis_cmplx_perorm(ret_dic[kv][-1], ret_dic[kv][0:5])
print('------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------')
print('                                               For Test set:')
print('K value: {:d} '.format(best_k))
ac, sen, pv,  tnr, f1, cm = knn(dis_dic, data_np, model, best_k, 1, tst_idx, trn_idx, cmret=True)
dis_cmplx_perorm(cm, list([ac, sen, pv, tnr, f1]))
print('------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------')
