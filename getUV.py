#getUV.py
import numpy as np
import prob2utils_with_bias as util
import pkl_help as pk
import projections

def train_tuning(data, K, reg, eta):
    
    col1 = data[:, 0]
    col2 = data[:, 1]
    col3 = data[:, 2]
    data = data.astype(int)

    M = int(max(col1))
    N = int(max(col2))
    result = util.train_model(943, 1682, K, eta, reg, data)
    return result

def tuning():
    data = np.loadtxt("data.txt")
    np.random.shuffle(data)
    training =  np.delete(data, np.s_[0:30000], axis=0)
    test = data[np.s_[0:30000], :]
    test = test.astype(int)
    best =  (-1, -1, -1)
    training_error = []
    test_error = []
    best_eout = [-1, -1, 1000]
    for k in [10, 20, 30, 50, 100, 200]:
        training_error.append([])
        test_error.append([])
        for r in [.05, .03, .07, .1, .15, .2, .3, .4, .5]:
            # training_error.append([])
            # test_error.append([])
            trained = train_tuning(training, k, r, .01)
            # print trained
            training_error[-1].append(trained[2])
            test_error[-1].append(util.get_err(trained[0], trained[1], test))
            print "K = ", k, ". Reg = ", r, "."
            print "Training Error: ", training_error[-1][-1]
            print "Test Error: ", test_error[-1][-1]
            print 
            if test_error[-1][-1] < best_eout[2]:
                best_eout = [k, r, test_error[-1][-1]]
    print best_eout

def gettingUV():
    data = np.loadtxt("data.txt")
    result = train_tuning(data, 20, .1, .01)
    return (result[0], result[1])
def gettingUV_from_pkl(file1, file2):
    U = pk.get_pkl(file1)
    V = pk.get_pkl(file2)
    return (U, V)

def get_proj_U_V(U, V):
    U_proj, V_proj = projections.get_projections(U, V)
    return (U_proj, V_proj)
def making_UVpkl(file1, file2):
    UV = gettingUV()
    U = UV[0]
    V = UV[1]
    pk.make_pkl(file1, U)
    pk.make_pkl(file2, V)

if __name__ == "__main__":

    making_UVpkl("saved_objs/U_k_20_withbias", "saved_objs/V_k_20_withbias")
    UV = gettingUV_from_pkl("saved_objs/U_k_20", "saved_objs/V_k_20")
    U = UV[0]
    V = UV[1]
    U_proj, V_proj = get_proj_U_V(U, V)
    print U_proj
    print V_proj