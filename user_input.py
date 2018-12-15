import sys
from DisplayMethods import show_array

def generate_col_headers(num_col, alphabet=False, verbose=False, head_arry=None):
    headers = list()

    if head_arry is not None:
        for i in range(2,len(head_arry)):
            headers.append(sys.argv[i])
    elif alphabet:
        cnt = 0
        for i in range(num_col):
            if i != 0 and i%26 == 0:
                cnt += 1
            if cnt == 0:
                headers.append(chr(ord('A')+i%26))
            if cnt > 0:
                headers.append(chr(ord('A')+i%26) + str(cnt))

    else:
        for i in range(num_col):
            headers.append('col '+str(i))

    if verbose:
        show_array(headers)

    return headers


def process_yes_no(depth_limit=None, impurity_threshold=None):

    t = (True,)
    f = (False,)

    while True:
        ans = input('Do you want to run a auto run with no tree limitation and entropy impurity ? (y/n)').lower()
        if ans == 'y':
            if depth_limit is not None:
                return True, depth_limit, 'd'
            elif impurity_threshold is not None:
                return True, impurity_threshold, 'i'
            else:
                return t
        elif ans == 'n':
            return f
        else:
            print('Your options are y or n')


# used to get how many test runs to perform
def process_num_input(prompt=None, error=None, low_lim=None, high_lim=None):
    while True:
        if prompt is None:
            test_num = int(input("how many test runs do you want to run? "))
        else:
            test_num = int(input(prompt))
        if low_lim is None or high_lim is None:
            if test_num < 0 or test_num > 200:
                if error is None:
                    print('The number of test runs must be greater than 0 and no more than 200.')
                else:
                    print(error)
            else:
                break
        elif low_lim is not None and high_lim is not None:
            if test_num < low_lim or test_num > high_lim:
                if error is None:
                    print('The number of test runs must be greater than 0 and no more than 200.')
                else:
                    print(error)
            else:
                break
    return test_num


def process_limit_type():

    while True:
        limit_type = int(input('What type of tree limitation would you like to use?:\n'
                               '                                        1) depth limit\n'
                               '                                        2) impurity threshold  '))
        if limit_type == 1:
            imp_thrsh = None
            depth_l = int(input('What depth limit would you like to use?:\n'
                                '                                        1) enter -1 for a range 2->10\n'
                                '                                        2) some  10 >= number >=2  '))
            if depth_l == -1:
                depth_limit = list(range(2, 11))
                break
            elif 10 >= depth_l >= 2:

                depth_limit = list([depth_l])
                break
            else:
                print('depth limit must be one of the allowed choices, no option for {:d}'.format(depth_l))
        elif limit_type == 2:
            depth_limit = None
            imp_l = int(input('What impurity threshold would you like to use?:\n'
                              '                                        1) enter -1 for a range .1->.9\n'
                              '                                        2) some  .9 >= number >=.1  '))
            if imp_l == -1:
                imp_thrsh = list()
                # for i in range(1,10):
                strt = 20
                for mul in range(0,2):
                    for i in range(strt, 0, -1):
                        imp_thrsh.append(i / (100*(10**mul)))
                    strt = 9
                break
            elif .20 >= imp_l >= .001:
                imp_thrsh.append(imp_l)
                break
            else:
                print('Threshold must be between .2 and .001 inclusive')
        else:
            print('There is no option {:d}'.format(limit_type))

    return limit_type, depth_limit, imp_thrsh


def process_error_type():
    while True:
        er_idx = int(input('What type of impurity test would you like to use?\n'
                           '                                                  1) entropy\n'
                           '                                                  2) gini index\n'
                           '                                                  3) missclassification error '))
        #if er_idx == 1 or er_idx == 2 or er_idx == 3:
        if 0 < er_idx <= 3:
            er_idx -= 1
            break
        else:
            print('There is no option {:d}'.format(er_idx))

    return er_idx