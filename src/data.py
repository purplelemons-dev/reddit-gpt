
from json import load, dump
#import multiprocessing as mp
import numpy as np

def write_data(data:dict):
    with open("resources/train.json", "w") as f:
        dump(data, f)

def read_data() -> dict:
    with open("resources/train.json", "r") as f:
        return load(f)
    
def read_primes() -> list:
    with open("resources/primes.json", "r") as f:
        data = load(f)
        return set(data), data[-1]

def to_base256(x:int) -> list[int]:
    result = [0] * 4
    for i in range(3, -1, -1):
        result[i] = x % 256
        x //= 256
    return result

# open "./resources/train_x.npy" and return the last value
def get_last_train_x() -> int:
    data = np.load("resources/train_x.npy")
    return data[-1]
#print(get_last_train_x())

def modify(chunk:np.ndarray):
    return np.array(to_base256(chunk[0]), dtype=np.uint8)

def main():
    for file in ("train_x", "test_x"):
        data:np.ndarray = np.load(f"resources/{file}.npy")
        new_data = []
        
        #zeros = np.zeros((data.shape[0], 3), dtype=np.uint8)
        #data = np.hstack((data, zeros))
        # the shape is (n,4)
        #np_modify = np.vectorize(to_base256, signature="(n)->(n)")
        #data = np_modify(data)
        data = np.apply_along_axis(modify, 1, data)
        data = data.astype(np.uint8)
        np.save(f"resources/{file}.npy", data)
        del data
#main()
#print(get_last_train_x())
#test=np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.uint16)
#print(test)
#test = test.reshape(test.shape[0], 1)
#print(test)
#zeros = np.zeros((test.shape[0], 3), dtype=np.uint16)
#test = np.hstack((test, zeros))
#print(test)
#test = np.apply_along_axis(modify, 1, test)
#print(test)

data:np.ndarray = np.load("resources/train_x.npy")
print(data[:16])
print(data[:8])
print(data[-8:])

#primes, last_prime = read_primes()
#print("done read primes")
#
#def check(x:int):
#    if not x%1000000:
#        print(x, flush=True)
#    return int(x in primes)
#
#if __name__ == "__main__":
#
#    # input uint32, output uint8/bool
#    train_x,train_x2 = set(range(1,last_prime+1)), list(range(1,last_prime+1))
#    print("done train x")
#
#    with mp.Pool(7) as p:
#        try:
#            train_y = list(p.imap_unordered(check, train_x, chunksize=10000000))
#        except KeyboardInterrupt:
#            p.terminate()
#            p.join()
#            raise KeyboardInterrupt
#    print(train_y[:16])
#    print("done train y")
#
#    write_data({"train_x":train_x2, "train_y":train_y})
