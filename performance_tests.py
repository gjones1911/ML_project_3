
def calculate_performance(cm, verbose=True, k=2):

    tp = float(cm[1][1])
    tn = float(cm[0][0])
    fp = float(cm[0][1])
    fn = float(cm[1][0])

    total = tp+tn+fp+fn

    if total != 0:
        accr = (tp + tn)/total
    else:
        accr = 0

    if tp+fn != 0:
        sens = tp/(tp+fn)
    else:
        sens = 0

    tsum = tp+fp
    if tsum > 0:
        ppv = tp/(tp+fp)
    else:
        ppv = 0

    tsum = tn + fp
    if tsum > 0:
        tnr = tn/(tn+fp)
    else:
        tnr = 0


    if ppv+sens != 0:
        f1 = (2*ppv*sens)/(ppv+sens)
    else:
        f1 = 0

    #if verbose:
    #    display_confustion_matrix_tf(tn,tp,fn,fp)
    #    display_knn_performance(list([accr, sens, ppv, tnr, f1]), k)

    if verbose:
        return tp, fp, fn, tn, accr, sens, ppv, tnr, f1
    else:
        return accr, sens, ppv, tnr, f1


def confusion_matrix(predictions, actual):

    tp = 0     # if classed malignant and is
    tn = 0     # if classed benign and is
    fp = 0     # if classed malignant and isn't
    fn = 0     # if classed benign and isn't

    #    predicted
    #          b    m
    # true: ben[tn, fp]
    #       mal[fn, tp]

    for act, pred in zip(actual,predictions):

        # if voted as benign
        if pred == 2:
            # it it is benign
            if act == 2:
                tn = tn + 1
            else:
                fn = fn + 1
        # if i predict it as malignant
        elif pred == 4:
            # it it is malignant
            if act == 4:
                tp = tp + 1
            else:
                fp = fp + 1

    cm = [[tn, fp],
          [fn, tp]]

    # show_array(cm)

    # confusion_matrix(cm, verbose=False)

    accr, sens, ppv, tnr, f1 = calculate_performance(cm, verbose=False)

    performance_array = list([accr, sens, ppv, tnr, f1])

    return cm, performance_array
    #return cm
