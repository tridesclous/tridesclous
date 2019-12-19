from tridesclous.dip import diptest
import numpy as np

import time


def test_diptest():
    
    n = 4000

    data = np.concatenate([np.random.randn(n)-3, np.random.randn(n)+3])
    
    t1 = time.perf_counter()
    pval = diptest(data)
    t2 = time.perf_counter()
    print('diptest', t2-t1)    

    #~ assert pval < 0.001
    
    print(pval)
    
    
    data = np.concatenate([np.random.randn(n)-3])
    t1 = time.perf_counter()
    pval = diptest(data)
    t2 = time.perf_counter()
    print('diptest', t2-t1)    
    
    #~ assert pval > 0.2
    
    print(pval)




if __name__ == '__main__':
    
    test_diptest()


