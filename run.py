import numpy as np
import importlib
import os
import sys

from scipy.linalg import svd
import time as rtime
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, davies_bouldin_score

from mpfcm import run_MPFCM, pred_cluster
from mfcm import run_MFCM
from mylib import compute_multiview_similarity, dual_anchor_graph_similarity, calc_representation, dual_anchor_graph_multiview, calculate_similarity, validity
from pso_mpfcm import pso_mpfcm
from config import purity_score, N_COMPONENT, clustering_accuracy
from MV_DAGL import MV_DAGL
from sklearn.decomposition import PCA
from augmented_matrix import create_augmented_matrix
from ELMSC.elmsc import elmsc
from ELMSC.augmented_data_matrix import compute_augmented_data_matrix
from ELMSC.utils import get_Aff

from normalization import normalize_multiview_data, normalize_multiview_data2, normalize_multiview_data3
# import ACM,ALOI,BBCSport,Caltech101,CiteSeer,Cora,Handwritten,Leaves100,MNIST4,MNIST10k,Movies,MSRCv5,NUSWIDE, OutdoorScene, Prokaryotic, ProteinFold, Reuters1200,Reuters1500,Sources3,UCI,WebKB,Wikipedia,Yale,Wikipediatest
PATH = os.path.dirname(os.path.abspath('mpfcm_8_6_12'))
MODULE_PATH = PATH + '/data/'
print(MODULE_PATH)
modules = [
    ("ACM", 3), 
    ("ALOI", 100), 
    # ("BBCSport", 5), 
    # ("Caltech", 102),
    # ("CiteSeer", 6), 
    # ("Cora", 7), 
    # ("Handwritten", 10),
    # ("mul_ORL", 40), 
    # ("MNIST4", 4), 
    # ("MNIST10k", 10),
    # ("Movies", 17), 
    ("MSRC-v5", 7), 
    # ("NUS-WIDE31", 12),
    # ("OutdoorScene", 8), 
    # ("Prokaryotic", 4), 
    # ("ProteinFold", 27),
    # ("Reuters-1200", 6), 
    ("3Sources", 6),
    # ("UCI", 10), 
    # ("WebKB", 2), 
    # ("Wikipedia", 10),
    ("Yale", 15), 
    # ("Wikipedia-test", 10), 
    ("Caltech101-20", 20)
    #, ("Reuters-1500", 12)
]

def checkDimen():
    for module_name, num_clusters in modules:
        print("run",module_name)
        try:
            # Dynamically import the module
            sys.path.append(MODULE_PATH)
            module = importlib.import_module(module_name)
            
            if hasattr(module, "X"):
                if module_name in ["ALOI","3Sources"]:
                    data = module.X
                else:
                    data = module.X[0]
                # print(data.shape)
                # data = normalize_multiview_data3(data)
                # print(data.shape)
                label = module.y
                c = num_clusters
                L = len(data)
                N = len(data[0])
                n_anchors = 4*c
                print(module_name)
                print(L,c,N)
                for l in range(L):
                    print(len(data[l][0]))
        except ModuleNotFoundError:
                print(f"Module {module_name} not found.")

def run(filename):
    for module_name, num_clusters in modules:
        if module_name == filename:
            print("run ",filename)
            try:
                # Dynamically import the module
                sys.path.append(MODULE_PATH)
                module = importlib.import_module(module_name)
                
                if hasattr(module, "X"):
                    if module_name in ["ALOI","3Sources"]:
                        data = module.X
                    else:
                        data = module.X[0]

                    # print(data)
                    label = np.array(module.y)
                    c = num_clusters
                    L = len(data)
                    N = len(data[0])
                    n_anchors = 4*c
                    
                    # if module_name == 'Yale':
                    #     dimen = [9,50,512]
                    #     for l in range(3):
                    #         dat = np.zeros((N,dimen[l]))
                    #         for k in range(N):
                    #             for i in range(dimen[l]):
                    #                 dat[k][i] = data[l][k][i]
                    #         data[l] = dat
                            
                    # dat = []
                    # if module_name == "MSRC-v5":
                    #     for l in range(L):
                    #         if l != 1:
                    #             dat.append(np.copy(data[l]))
                    # L -= 1
                    # data = dat
                    print(L,c,N)
                    for l in range(L):
                        print("dimension of view " + str(l) + ":" + str(len(data[l][0])))

                    # print("Begin run:")                    
                    # # reduce each 
                    # for l in range(L):
                    #     pca = PCA(n_components=c)
                    #     data[l] = pca.fit_transform(data[l])

                    data = normalize_multiview_data3(data)

                    # alpha=0.01
                    # beta =alpha/(N * L)
                    # sigma= 0.05
                    # omega = 0.9
                    # sump = 1 - alpha - beta - sigma - omega
                    # theta = sump
                    # # theta = 0
                    
                    # ro = 0

                    alpha=10
                    beta =alpha/(N * L)
                    sigma= 10
                    omega = 1
                    sump = 1 - alpha - beta - sigma - omega
                    theta = 0.1
                    ro = 1

                    params = {
                        "alpha": alpha,
                        "beta" : beta,
                        "sigma" : sigma,
                        "theta" : theta,
                        "omega" : omega,
                        "ro"    : ro
                    }
                    print("calculate augmented_matrix")
                    Aff = []
                    augmented_matrix = compute_augmented_data_matrix(data)
                    K =  100
                    lamb = 1e-2

                    data_elmsc = []
                    for l in range(L):
                        data_elmsc.append(data[l].T)                    

                    # Run ELMSC
                    print("run elmsc")
                    Z, Z_a, obj_val, H_a = elmsc(data_elmsc, augmented_matrix, K, lamb, 10)
                    Aff = get_Aff(Z)
                    print("Affinity matrix")
                    print(Aff.shape)

                    # Chạy thuật toán MV_DAGL
                    print("Begin run MV_DAGL:")
                    z_c, z_list = MV_DAGL(data, n_anchors, max_iter=100, alpha=0.01, tol=1e-6)

                    data_new = []    
                    # print("Begin calculate similarity:")
                    similarity = []
                    for l in range(L):
                        tg = np.array(z_list[l].T @ z_list[l])
                        # normalizedData = (tg - np.min(tg))/(np.max(tg)-np.min(tg))
                        similarity.append(tg)
                        U, _, _ = svd(z_list[l].T, full_matrices=False)                        
                        data_new.append(U[:, :n_anchors])

                    U, _, _ = svd(z_c.T, full_matrices=False)
                    data_new.append(U[:, :n_anchors])

                    tg = np.array(z_c.T @ z_c)
                    # normalizedData = (tg-np.min(tg))/(np.max(tg)-np.min(tg))
                    similarity.append(tg) 
            
                    runtime, cluster_predict = run_MPFCM(L, c, N, data_new, params, similarity, Aff, label)
                                            
                else:
                    print(f"Module {module_name} does not have attribute 'X'.")
            except ModuleNotFoundError:
                print(f"Module {module_name} not found.")

# checkDimen()
# run("ALOI")
run("Yale")
# run_pso("Yale")
# run_pso("MSRC-v5")
# run("3Sources")
# run_test("MSRC-v5")
# run_pso("MSRC-v5")
# run_test("Caltech101-20")
# run_pso("Yale")
# run_pso("3Sources")
# run_fcm("Yale")