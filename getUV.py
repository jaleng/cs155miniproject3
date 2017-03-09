#getUV.py
import numpy as np
import prob2utils as util
import pkl_help as pk


def train_tuning(data, K, reg, eta):
    
    col1 = data[:, 0]
    col2 = data[:, 1]
    col3 = data[:, 2]
    data = data.astype(int)


    M = int(max(col1))
    N = int(max(col2))
    result = util.train_model(M, N, K, eta, reg, data)
    return result

def tuning():
    data = np.loadtxt("data.txt")
    training =  np.delete(data, np.s_[0:30000], axis=0)
    test = data[np.s_[0:30000], :]
    test = test.astype(int)
    best =  (-1, -1, -1)
    training_error = []
    test_error = []
    for k in [10, 20, 30, 50, 100, 200]:
        training_error.append([])
        test_error.append([])
        for e in [1e-3, 1e-2, 1e-1, 1, 10, 100]:
            training_error[-1].append([])
            test_error[-1].append([])
            for r in [.001, .01, .1, 1, 10, 20]:
                # training_error.append([])
                # test_error.append([])
                trained = train_tuning(training, k, r, e)
                # print trained
                training_error[-1][-1].append(trained[2])
                test_error[-1][-1].append(util.get_err(trained[0], trained[1], test))
                print "K = ", k, ". Eta = ", e, ". Reg = ", r, ".\n"
                print "Training Error: ", training_error[-1][-1][-1]
                print "Test Error: ", test_error[-1][-1][-1]
                print 



if __name__ == "__main__":
    # data = np.loadtxt("data.txt")
    # print len(data)
    tuning()