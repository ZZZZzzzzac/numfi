from refactor import *

if __name__ == '__main__':
    x = numfi([1,2,3,4],1,16,8)
    x_plus_256 = x + np.int64([256])
    print('d')