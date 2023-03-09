
from time import perf_counter
import numpy as np
from math import sqrt

def isPrime(n:int)->bool:
    if n == 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(sqrt(n))+1, 2):
        if n % i == 0:
            return False
    return True


if __name__=="__main__":
    st = perf_counter()
    #for i in ("primes1.txt","primes2.txt","primes3.txt","primes4.txt","primes5.txt","primes6.txt","primes7.txt","primes8.txt","primes9.txt","primes10.txt"):
    #    with open("resources/"+i, "r") as f:
    #        for x in f.readlines()[1:]:
    #            for x in x.strip().replace("\n","").split(" "):
    #                try:
    #                    primes += [int(x)]
    #                except ValueError:
    #                    pass
    primes=[]
    with open("resources/primes11.txt", "r") as f:
        for x in f.readlines()[1:]:
            for x in x.strip().replace("\n","").split(" "):
                try:
                    primes.append(int(x))
                except ValueError:
                    pass

    first, last = primes[0], primes[-1]
    primes=set(primes)
    test_x, test_y = list(range(first, last+1)), []
    for i in test_x:
        test_y.append(1 if i in primes else 0)
    np.save("resources/test_x.npy", np.array(test_x, dtype=np.uint32))
    np.save("resources/test_y.npy", np.array(test_y, dtype=np.uint8))
    
    print(perf_counter()-st)

