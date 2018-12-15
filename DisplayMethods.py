
def show_array(array):
    for row in array:
        print(row)
    return


def show_array_selection(array, selection):
    for row in selection:
        print('row:',row)
        print(array[row])
    return





def display_confustion_matrix_tf(tn,tp, fn, fp):
    print('           Confusion Matrix:')
    print(' ___________________________________')
    print('|              |   Predicted        |')
    print('|  True Class  |____________________|')
    print('|______________|_benign_|_malignant_|')
    print('|__benign______|__{:_>4d}__|___{:_>4d}____|'.format(int(tn), int(fp)))
    print('|__malignant___|__{:_>4d}__|___{:_>4d}____|'.format(int(fn), int(tp)))
    print('')
    return


def display_knn_performance(p_array, k_val=2):
    print('')
    #print('Averages for k value of {:d}'.format(int(k_val)))
    print('Accuracy : {:f}'.format(p_array[0]))
    print('Sensitivity : {:f}'.format(float(p_array[1])))
    print('Precision : {:f}'.format(float(p_array[2])))
    print('True Negative Rate : {:f}'.format(float(p_array[3])))
    print('F1 score : {:f}'.format(float(p_array[4])))
    print('')
    return


def dis_cmplx_perorm(cm, p_array):
    display_confustion_matrix_tf(cm[0][0], cm[1][1], cm[1][0], cm[0][1])
    display_knn_performance(p_array)
    return