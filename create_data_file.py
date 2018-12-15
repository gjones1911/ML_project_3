from DataCleaner import create_data_file_from_file
from DataCleaner import multi_strip
import sys

if len(sys.argv) == 1:
    c_file = input('Give me:  the name of the data file you want to create or enter standard to use the standard name: ')

    c_file = multi_strip(c_file, ' ')

    if c_file != 'standard':
        create_data_file_from_file(input('Give me the name of the file you want to use: '), created_file=c_file)
    else:
        create_data_file_from_file(input('Give me the name of the file you want to use: '))
elif len(sys.argv) == 2:
    create_data_file_from_file(sys.argv[1])
elif len(sys.argv) == 3:
    create_data_file_from_file(sys.argv[1], sys.argv[2])
elif len(sys.argv) == 4:
    create_data_file_from_file(sys.argv[1], sys.argv[2], )
