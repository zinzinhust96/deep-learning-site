data = [1,2,3,4,5,6,7,8,9]
for index, value in enumerate(list(zip(data[0::3], data[1::3], data[2::3]))):
    # print (str(i), '+', str(k), '=', str(i+k))
    # print (l)
    print(index)
    print(value)