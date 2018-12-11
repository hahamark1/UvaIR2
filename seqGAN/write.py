dict = {'a': 1, 'b': 2}


with open("iding.txt", "a") as myfile:
    myfile.write('Epoch: {}'.format('boe'))
    for key in dict.keys():
        myfile.write('{}: {}\n'.format(key, dict[key]))
