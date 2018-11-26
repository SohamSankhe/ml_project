import pymysql

HOST = "localhost"
USER = "root"
PASSWORD = "root123"
DB = "mlproject"

def getConnection(host = HOST, user = USER, password = PASSWORD, db = DB):
    # Connect to the database
    connection = pymysql.connect(host=host,
                                 user=user,
                                 password=password,
                                 db=db,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection

'''
# test
lst = [[1,2,3], [3,4,5]]

y = []
import numpy as np
x = np.zeros((lst.__len__(), lst[0].__len__() - 1))
i = 0
for l in lst:
    y.append(l[0])
    x[i,:] = l[1:]
    i += 1

print(y)
print(x)

'''