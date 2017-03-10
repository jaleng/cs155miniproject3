import numpy as np
import sys
import pkl_help as pk

def get_projections(U, V):
    A, Sigma, B = np.linalg.svd(V)
    U_projected = np.dot(A[:, 0:2].T, U.T)
    V_projected = np.dot(A[:, 0:2].T, V)

    # Normalize to make Unit Vectors
    # U_projected[0, :] = U_projected[0, :] / np.linalg.norm(U_projected, axis=1)[0]
    # U_projected[1, :] = U_projected[1, :] / np.linalg.norm(U_projected, axis=1)[1]
    # V_projected[0, :] = V_projected[0, :] / np.linalg.norm(V_projected, axis=1)[0]
    # V_projected[1, :] = V_projected[1, :] / np.linalg.norm(V_projected, axis=1)[1]
    
    return U_projected, V_projected

if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print("usage: python projections.py U_file V_file UV_projections_file")
    _, U_file, V_file, UV_projections_file = sys.argv
    U = pk.get_pkl(U_file)
    V = pk.get_pkl(V_file)
    U_proj, V_proj = get_projections(U, V)
    pk.make_pkl(UV_projections_file, (U_proj, V_proj))
