import datetime
import os
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import davies_bouldin_score, normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, davies_bouldin_score

import multiprocessing
import time
import sys


# Semi-supervised learning parameters
LABEL_RATIO = 0.5  # Ratio of labeled data (default: 50%)
RANDOM_SEED = 42   # Random seed for reproducibility

# Convergence parameters
EPSILON = 0.0001
EPSILON2 = 0.001
MAXSTEPS = 100

# Debug flag
ISDEBUG = True

# Print validity flag
printValidity = True

# Optimization parameters
ALPHA_XI = 0.1

N_COMPONENT = 10

NPAR = 20

from sys import exit

def init_w(L):
    """Initialize view weights"""
    w = np.ones(L) / L
    return w

def init_u_eta_xi(c, N):
    """Initialize membership, typicality and uncertainty matrices"""
    u = np.random.rand(c, N)
    eta = np.random.rand(c, N)
    xi = np.random.rand(c, N)
    
    # Normalize u
    for k in range(N):
        tg = 0
        for j in range(c):
            tg += u[j][k]
        for j in range(c):
            u[j][k] = u[j][k] / tg
            
    # Normalize eta
    for k in range(N):
        tg = 0
        for j in range(c):
            tg += eta[j][k]
        for j in range(c):
            eta[j][k] = eta[j][k] / tg
            
    # Normalize xi
    for k in range(N):
        tg = 0
        for j in range(c):
            tg += xi[j][k]
        for j in range(c):
            xi[j][k] = xi[j][k] / tg
            
    return u, eta, xi

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def clustering_accuracy(true_labels, pred_labels):
    """
    Hàm tính chỉ số Accuracy (ACC) cho phân cụm dữ liệu.
    
    Parameters:
    true_labels (array-like): Nhãn thực tế (ground truth).
    pred_labels (array-like): Nhãn phân cụm dự đoán.
    
    Returns:
    float: Chỉ số ACC (accuracy).
    """
    # Tạo ma trận confusion giữa các nhãn thực tế và nhãn phân cụm
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Tìm sự khớp tốt nhất giữa nhãn thực tế và nhãn phân cụm
    # Sử dụng thuật toán Hungarian (linear_sum_assignment) để tối ưu hóa sự ghép nối
    cost_matrix = -cm  # Sử dụng giá trị âm vì thuật toán tìm kiếm chi phí tối thiểu
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Tính toán số lượng các điểm phân loại chính xác (khớp với nhãn thực tế)
    correct_matches = cm[row_ind, col_ind].sum()
    
    # Tính toán Accuracy
    accuracy = correct_matches / len(true_labels)
    return accuracy

def printValidity(label,cluster_predict,opt):
    
    acc = clustering_accuracy(label,cluster_predict)
    ari = adjusted_rand_score(label,cluster_predict)
    f1 = f1_score(label, cluster_predict, average='micro')
    nmi = normalized_mutual_info_score(label,cluster_predict)
    purity = purity_score(label,cluster_predict)


    if opt == 1:
        print("ACC score view: ",acc)
        print("ARI score view: ",ari)
        print("F1 score view: ",f1)
        print("NMI score view: ",nmi)
        print("Purity score view: ",purity)

    return acc, ari, f1, nmi, purity