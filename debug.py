from refactor import *

if __name__ == '__main__':
    q = np.random.rand(3,2)
    s,w,f = 0,11,7
    x = numfi(q.copy(), s,w,f,fixed=True)
    y = x * 1.0123