dic =  {}
if 'a' not in dic.keys():
    print('123')
a_1 = [1,2,0]
a_2 = [2,3,1]
a_pos = []
a_pos.append(a_1)
a_pos.append(a_2)
dic['a'] = a_pos
print(dic['a'])
a_3 = [3,5,2]
dic['a'].append(a_3)
print(dic['a'])
