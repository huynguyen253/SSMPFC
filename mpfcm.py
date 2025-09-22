import time
import sys
import numpy as np
from mylib import calculate_fcm, calculate_fcm_all, validity
from scipy.spatial.distance import cdist
from config import ALPHA_XI,EPSILON,EPSILON2,MAXSTEPS,init_u_eta_xi,init_w, printValidity, ISDEBUG, RANDOM_SEED, LABEL_RATIO
#from normalization import normalize_multiview_data3
from sklearn.decomposition import PCA
from fcmeans import FCM

from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, davies_bouldin_score

def calc_D(L, c, N, v_s, rep):
    d_s = np.zeros((L + 1, c, N))
    for i in range(L + 1):
       d_s[i] = cdist(rep[i], v_s[i], metric='euclidean').T #(c,N)
    return d_s

def calculate_A_u(u, eta, xi, w, L, c, N, d_s, params, similarity, au_matrix, is_labeled=None, true_clusters=None):
    """Calculate A_kj matrix using label information as guidance - vectorized
    
    Công thức:
    - Với điểm CÓ NHÃN: A_kj = (α(1-ξ_kj)^2 + ω)∑_l=0^L w_l d_lkj + θ∑_l=0^L∑_i=1^N S_lki^S + ρ
    - Với điểm KHÔNG NHÃN: A_kj = α(1-ξ_kj)^2 ∑_l=0^L w_l d_lkj + θ∑_l=0^L∑_i=1^N S_lki^S + ρ
    """
    # Calculate weighted sum of distances across all views
    weighted_distances = np.zeros((c, N))
    for l in range(L + 1):
        weighted_distances += w[l] * d_s[l]
    
    # Calculate similarity term
    similarity_term = np.zeros(N)
    for l in range(L + 1):
        similarity_term += np.sum(similarity[l], axis=1)
    
    # Initialize A_kj matrix
    A = np.zeros((c, N))
    
    # Create masks for labeled and unlabeled data
    labeled_mask = np.zeros(N, dtype=bool)
    if is_labeled is not None and true_clusters is not None:
        labeled_mask = is_labeled & (true_clusters >= 0)
    unlabeled_mask = ~labeled_mask
    
    # Process unlabeled points - with only alpha term
    if np.any(unlabeled_mask):
        # Cho điểm KHÔNG NHÃN: A_kj = α(1-ξ_kj)^2 ∑_l=0^L w_l d_lkj + θ∑_l=0^L∑_i=1^N S_lki^S + ρ
        for j in range(c):
            # Term 1: α(1-ξ_kj)^2 ∑_l=0^L w_l d_lkj
            term1 = params["alpha"] * ((1 - xi[j, unlabeled_mask]) ** 2) * weighted_distances[j, unlabeled_mask]
            
            # Term 2: θ∑_l=0^L∑_i=1^N S_lki^S
            term2 = params["theta"] * similarity_term[unlabeled_mask]
            
            # Term 3: ρ
            term3 = params["rho"]
            
            # Combine terms
            A[j, unlabeled_mask] = term1 + term2 + term3
    
    # Process labeled points - with both alpha and omega terms
    if np.any(labeled_mask):
        # Cho điểm CÓ NHÃN: A_kj = (α(1-ξ_kj)^2 + ω)∑_l=0^L w_l d_lkj + θ∑_l=0^L∑_i=1^N S_lki^S + ρ
        for j in range(c):
            # Tính các trọng số
            alpha_weights = params["alpha"] * ((1 - xi[j, labeled_mask]) ** 2)
            
            # Term 1: (α(1-ξ_kj)^2 + ω)∑_l=0^L w_l d_lkj
            term1 = (alpha_weights + params["omega"]) * weighted_distances[j, labeled_mask]
            
            # Term 2: θ∑_l=0^L∑_i=1^N S_lki^S
            term2 = params["theta"] * similarity_term[labeled_mask]
            
            # Term 3: ρ
            term3 = params["rho"]
            
            # Combine terms
            A[j, labeled_mask] = term1 + term2 + term3
    
    return A

def calculate_Bu(u, eta, xi, A, L, is_labeled=None):
    """Calculate B_j using vectorized formula"""
    # Reshape for broadcasting
    A_expanded = A.reshape(A.shape[0], 1, A.shape[1])
    A_expanded_repeat = np.repeat(A.reshape(1, A.shape[0], A.shape[1]), A.shape[0], axis=0)
    
    # Calculate ratio using broadcasting
    ratio = A_expanded / np.maximum(A_expanded_repeat, 1e-10)  # Avoid division by zero
    
    # Sum along the last dimension (N)
    B = np.sum(ratio, axis=2)
    
    # Sum along cluster dimension for each j
    B = np.sum(B, axis=1)
    
    return B

def calculate_G_u(u, L, c, N, w, d_s, params, similarity, is_labeled=None, true_clusters=None):
    """Calculate G_kj matrix using label information as guidance - vectorized
    
    Công thức:
    - Với điểm CÓ NHÃN: G_kj = θ∑_l=0^L∑_i=1^N μ_ij S_lki^S + ω∑_l=0^L w_l f_kj d_lkj
    - Với điểm KHÔNG NHÃN: G_kj = θ∑_l=0^L∑_i=1^N μ_ij S_lki^S
    """
    G = np.zeros((c, N))
    
    # Create masks for labeled and unlabeled data
    labeled_mask = np.zeros(N, dtype=bool)
    if is_labeled is not None and true_clusters is not None:
        labeled_mask = is_labeled & (true_clusters >= 0)
    unlabeled_mask = ~labeled_mask
    
    # Calculate first term with similarity - common for both labeled and unlabeled points
    # θ∑_l=0^L∑_i=1^N μ_ij S_lki^S
    for j in range(c):
        # For each cluster j
        term1 = np.zeros(N)
        for l in range(L + 1):
            # For each view l
            # Matrix multiply: u[j, :] * similarity[l]
            term1 += np.matmul(similarity[l], u[j, :])
        
        G[j, :] += term1 * params["theta"]
    
    # Additional term for labeled points only
    # ω∑_l=0^L w_l f_kj d_lkj
    if np.any(labeled_mask):
        for j in range(c):
            # Create array for f_kj (1 if true cluster, 0 otherwise)
            f_kj = np.zeros(N)
            for k in np.where(labeled_mask)[0]:
                if j == true_clusters[k]:
                    f_kj[k] = 1.0
            
            # Calculate ω∑_l=0^L w_l f_kj d_lkj only for labeled points
            term2 = np.zeros(N)
            for l in range(L + 1):
                term2[labeled_mask] += w[l] * f_kj[labeled_mask] * d_s[l][j, labeled_mask]
            
            G[j, labeled_mask] += term2[labeled_mask] * params["omega"]
    
    return G

def calculate_u(u, xi, eta, w, L, c, N, d_s, params, similarity, au_matrix, is_labeled=None, true_clusters=None):
    '''
    Calculate membership values using label information as guidance - vectorized.
    '''
    A = calculate_A_u(u, eta, xi, w, L, c, N, d_s, params, similarity, au_matrix, is_labeled, true_clusters)
    B = calculate_Bu(u, eta, xi, A, L, is_labeled)
    G = calculate_G_u(u, L, c, N, w, d_s, params, similarity, is_labeled, true_clusters)
    
    # Calculate G/A ratio for each cluster and data point with safe division
    G_A_ratio = G / np.maximum(A, 1e-10)  # Avoid division by zero
    
    # Calculate sum of G/A ratios for each data point (sum over clusters)
    sum_G_A = np.sum(G_A_ratio, axis=0)
    
    # Calculate membership values vectorized
    # Reshape B for broadcasting
    B_reshaped = B.reshape(-1, 1)
    u_new = (1.0 / B_reshaped) + (G_A_ratio - sum_G_A / B_reshaped)
    
    # For labeled data: adjust membership values using label information
    if is_labeled is not None and true_clusters is not None:
        # Find labeled samples
        labeled_indices = np.where(is_labeled & (true_clusters >= 0))[0]
        
        if len(labeled_indices) > 0:
            label_weight = 0.8  # Weight for label influence
            
            # Process all labeled samples at once
            for idx in labeled_indices:
                true_label = true_clusters[idx]
                
                # Create mask for true and false clusters
                cluster_mask = np.arange(c) == true_label
                
                # Apply weight adjustments
                u_new[cluster_mask, idx] = u_new[cluster_mask, idx] * (1 - label_weight) + label_weight
                u_new[~cluster_mask, idx] = u_new[~cluster_mask, idx] * (1 - label_weight)
                
                # Normalize to ensure sum = 1
                u_new[:, idx] /= np.sum(u_new[:, idx])
    
    # Ensure constraints
    u_new = np.clip(u_new, EPSILON, 1 - EPSILON)
    
    return u_new

def calculate_eta(d_s, u, eta, xi, L, c, N, params):
    """Calculate η_kj according to new formula - vectorized
    η_kj = e^(ξ_kj)/(L∑_(j=1)^C e^(ξ_kj))
    """
    # Calculate e^(ξ_kj) for all j,k
    exp_xi = np.exp(np.clip(xi, -500, 500))  # Clip to avoid overflow
    
    # Calculate denominator: L∑_(j=1)^C e^(ξ_kj) for each k
    denominator = L * np.sum(exp_xi, axis=0)
    
    # Calculate η_kj = e^(ξ_kj)/(L∑_(j=1)^C e^(ξ_kj)) using broadcasting
    eta_new = exp_xi / np.maximum(denominator, 1e-10)  # Avoid division by zero
    
    # Ensure constraint
    eta_new = np.maximum(eta_new, EPSILON)
    
    return eta_new

# calculate E_xi - vectorized
def calculate_E_xi(u, L, c, N, w, d_s, params):
    """Calculate E_kj according to new formula - vectorized
    E_kj = 2αμ_kj^2 ∑_(l=0)^L w_l d_lkj
    """
    # Calculate weighted sum of distances across all views
    weighted_sum = np.zeros((c, N))
    for l in range(L + 1):
        weighted_sum += w[l] * d_s[l]
    
    # Calculate E_kj = 2αμ_kj^2 * weighted_sum using broadcasting
    E = 2 * params["alpha"] * (u ** 2) * weighted_sum
    
    return E

# for calculate xi - vectorized
def calculate_xi(u, eta, w, L, c, N, d_s, params):
    """Calculate ξ_kj according to new formula - vectorized
    ξ_kj = 1 - (βη_kj)/E_kj + (1/L - C + β∑_(i=1)^C(η_ki/E_ki))/(∑_(i=1)^C(E_kj/E_ki))
    """
    # Calculate E matrix
    E = calculate_E_xi(u, L, c, N, w, d_s, params)
    
    # Add small epsilon to avoid division by zero
    E_safe = np.maximum(E, 1e-10)
    
    # First term: 1 - (βη_kj)/E_kj - using broadcasting
    term1 = 1 - (params["beta"] * eta) / E_safe
    
    # Second term calculation
    
    # Calculate η_ki/E_ki for all i,k
    eta_E = eta / E_safe
    
    # Sum over clusters: ∑_(i=1)^C(η_ki/E_ki) for each k
    # This gives a vector of length N
    sum_eta_E = np.sum(eta_E, axis=0)
    
    # For each j and k, calculate ∑_(i=1)^C(E_kj/E_ki)
    # Need to reshape for broadcasting
    E_reshaped_1 = E.reshape(c, 1, N)  # For numerator: E_kj
    E_reshaped_2 = E.reshape(1, c, N)  # For denominator: E_ki
    
    # Calculate E_kj/E_ki for all j,i,k
    E_ratio = E_reshaped_1 / np.maximum(E_reshaped_2, 1e-10)
    
    # Sum over i: ∑_(i=1)^C(E_kj/E_ki) for each j,k
    sum_E_ratio = np.sum(E_ratio, axis=1)  # Shape: (c, N)
    
    # Calculate numerator: (1/L - c + β∑_(i=1)^C(η_ki/E_ki)) for each k
    numerator = (1/L - c + params["beta"] * sum_eta_E)
    
    # Reshape for broadcasting
    numerator_reshaped = numerator.reshape(1, N)
    
    # Calculate second term using broadcasting
    term2 = numerator_reshaped / np.maximum(sum_E_ratio, 1e-10)
    
    # Calculate final xi
    xi_new = term1 + term2
    
    # Ensure constraints
    xi_new = np.clip(xi_new, EPSILON, 1 - EPSILON)
    
    return xi_new

def recalc_u_eta_xi(u, eta, xi, L, c, N):
    """Recalculate and normalize u, eta, xi matrices - vectorized"""
    # Clip values to ensure constraints
    u = np.clip(u, EPSILON, 1 - EPSILON)
    eta = np.clip(eta, EPSILON, 1 - EPSILON)
    xi = np.clip(xi, EPSILON, 1 - EPSILON)
    
    # Calculate sum of u, eta, xi for each j,k
    total_sum = u + eta + xi
    
    # Create mask for where sum > 1
    mask = total_sum > 1
    
    # Only normalize where sum > 1
    if np.any(mask):
        # Reshape for broadcasting if needed
        sum_reshaped = np.where(mask, total_sum, 1.0)
        
        # Normalize each component
        u = np.where(mask, u / sum_reshaped, u)
        eta = np.where(mask, eta / sum_reshaped, eta)
        xi = np.where(mask, xi / sum_reshaped, xi)
    
    return u, eta, xi

def calculate_V(u, eta, xi, rep, L, c, N, params, is_labeled=None, true_clusters=None):
    """Calculate cluster centers for each view - vectorized
    
    Công thức:
    V_lj = (α∑_(k=1)^N▒〖((1-ξ_kj)μ_kj)^2 F_lk 〗+ω∑_(k=1)^(N_L)▒〖(μ_kj-f_kj)^2 F_lk 〗)/(α∑_(k=1)^N▒((1-ξ_kj)μ_kj)^2 +ω∑_(k=1)^(N_L)▒(μ_kj-f_kj)^2)
    
    Trong đó:
    - F_lk là vector đặc trưng của điểm k trong view l
    - f_kj = 1 nếu điểm k thuộc cụm j (trong ground truth), f_kj = 0 nếu ngược lại
    - N_L là tập hợp các điểm có nhãn
    """
    v = []  # To store V_ij for each view
    
    # Create mask for labeled points (N_L)
    labeled_mask = np.zeros(N, dtype=bool)
    if is_labeled is not None and true_clusters is not None:
        labeled_mask = is_labeled & (true_clusters >= 0)
    
    for l in range(L + 1):
        feature_dim = len(rep[l][0])
        vl = np.zeros((c, feature_dim))
        
        for j in range(c):
            # Khởi tạo tử số và mẫu số
            numerator = np.zeros(feature_dim)
            denominator = 0
            
            # Tính toán thành phần đầu tiên: α∑((1-ξ_kj)μ_kj)^2 F_lk
            # Áp dụng cho tất cả các điểm
            for k in range(N):
                weight = params["alpha"] * ((1 - xi[j][k]) * u[j][k])**2
                # Cộng dồn vào tử số
                numerator += weight * rep[l][k]
                # Cộng dồn vào mẫu số (không nhân với F_lk)
                denominator += weight
            
            # Tính toán thành phần thứ hai: ω∑(μ_kj-f_kj)^2 F_lk
            # Chỉ áp dụng cho các điểm có nhãn
            if np.any(labeled_mask):
                for k in np.where(labeled_mask)[0]:
                    # f_kj = 1 nếu j là cụm thật của k, ngược lại = 0
                    f_kj = 1.0 if true_clusters[k] == j else 0.0
                    # Tính (μ_kj-f_kj)^2
                    diff_squared = (u[j][k] - f_kj)**2
                    # Trọng số cho thành phần này
                    weight = params["omega"] * diff_squared
                    # Cộng dồn vào tử số
                    numerator += weight * rep[l][k]
                    # Cộng dồn vào mẫu số (không nhân với F_lk)
                    denominator += weight
            
            # Tính V_lj = numerator / denominator
            if denominator > 1e-10:
                vl[j] = numerator / denominator
            else:
                # Fallback nếu mẫu số quá nhỏ
                vl[j] = np.mean(rep[l], axis=0)
                
        v.append(vl)
    
    return v

def calculate_w(w, d_s, u, eta, xi, L, c, N, params, similarity, is_labeled=None, true_clusters=None):
    """Calculate w_l using different formulas - vectorized:
    
    Công thức:
    - View 0 (với nhãn): K_l = exp((α∑_k=1^N∑_j=1^C((1-ξ_kj)μ_kj)^2 d_lkj + ω∑_j=1^C∑_k=1^N_L(μ_kj-f_kj)^2 d_lkj)/σ)
    - View khác (không nhãn): K_l = exp((α∑_k=1^N∑_j=1^C((1-ξ_kj)μ_kj)^2 d_lkj)/σ)
    """
    K = np.zeros(L + 1)
    
    # Calculate ((1-ξ_kj)μ_kj)^2 for all j,k once
    modifier = ((1 - xi) * u) ** 2
    
    # Create mask for labeled points
    labeled_mask = np.zeros(N, dtype=bool)
    if is_labeled is not None and true_clusters is not None:
        labeled_mask = is_labeled & (true_clusters >= 0)
    
    # Calculate K_l for each view
    for l in range(L + 1):
        # First term (common for all views) - vectorized
        # α∑_k=1^N∑_j=1^C((1-ξ_kj)μ_kj)^2 d_lkj
        term1 = params["alpha"] * np.sum(modifier * d_s[l])
        
        # Second term (chỉ cho view với nhãn)
        term2 = 0.0
        
        # Nếu có điểm có nhãn
        if np.any(labeled_mask):
            # Tính f_kj và (μ_kj-f_kj)^2 d_lkj
            labeled_points_sum = 0.0
            
            for j in range(c):
                # Khởi tạo f_kj (1 nếu j là cụm thực, 0 nếu ngược lại)
                f_kj = np.zeros(N)
                for k in np.where(labeled_mask)[0]:
                    if j == true_clusters[k]:
                        f_kj[k] = 1.0
                
                # Tính (μ_kj-f_kj)^2 d_lkj cho điểm có nhãn
                diff_squared = (u[j, labeled_mask] - f_kj[labeled_mask]) ** 2
                labeled_points_sum += np.sum(diff_squared * d_s[l][j, labeled_mask])
            
            # Nhân với ω cho term2
            term2 = params["omega"] * labeled_points_sum
        
        # Calculate K_l with overflow protection
        # For view 0, include both term1 and term2
        # For other views, use only term1
        if l == 0:  # View 0 - với nhãn
            exp_val = (term1 + term2) / max(params["sigma"], 1e-10)
        else:       # View khác - không nhãn
            exp_val = term1 / max(params["sigma"], 1e-10)
        
        # Clip exp_val to prevent overflow
        exp_val = np.clip(exp_val, -500, 500)
        
        # Calculate K_l
        K[l] = np.exp(exp_val)
    
    # Normalize K values to prevent overflow
    max_K = np.max(K)
    if max_K > 0:
        K = K / max_K
    
    # Add small epsilon to prevent division by zero
    K = np.maximum(K, 1e-10)
    
    # Calculate weights - vectorized
    # For each l, calculate sum_K_ratio = ∑_(h=0)^L K_h/K_l
    # Using broadcasting to calculate all ratios at once
    K_ratios = K.reshape(1, -1) / K.reshape(-1, 1)  # Shape: (L+1, L+1)
    sum_K_ratio = np.sum(K_ratios, axis=1)  # Sum along columns
    
    # Calculate w_l = 1/sum_K_ratio for all l
    w_new = 1.0 / sum_K_ratio
    
    # Normalize to ensure sum of weights is 1
    w_sum = np.sum(w_new)
    if w_sum > 0:
        w_new = w_new / w_sum
    else:
        # If all weights are 0, use uniform weights
        w_new = np.ones(L + 1) / (L + 1)
    
    return w_new

def calc_F(u, eta, xi, w, L, c, N, d_s, params, similarity, au_matrix, is_labeled=None, true_clusters=None):
    """Calculate objective function F according to new formula - vectorized:
    
    F = α∑_(l=0)^L w_l ∑_(k=1)^N∑_(j=1)^C((1-ξ_kj)μ_kj)^2 d_lkj 
        - β∑_(k=1)^N∑_(j=1)^C n_kj(ln(n_kj)-ξ_kj)
        - σ∑_(l=0)^L w_l(ln(w_l))
        + θ∑_(l=0)^L∑_(j=1)^C∑_(k=1)^N∑_(i=1)^N S_lki^S(μ_kj-μ_ij)^2
        + ω∑_(l=0)^L w_l ∑_(j=1)^C∑_(k=1)^(N_L)(μ_kj-f_kj)^2 d_lkj
        + ρ∑_(k=1)^N∑_(j=1)^C μ_kj^2
    """
    # First term: α∑_(l=0)^L w_l ∑_(k=1)^N∑_(j=1)^C((1-ξ_kj)μ_kj)^2 d_lkj
    # Pre-compute ((1-ξ_kj)μ_kj)^2 for all j,k
    modifier = ((1 - xi) * u)**2
    
    f1 = 0
    for l in range(L + 1):
        # Vectorized sum: ∑_(k=1)^N∑_(j=1)^C((1-ξ_kj)μ_kj)^2 d_lkj
        f1 += w[l] * np.sum(modifier * d_s[l])
    f1 *= params["alpha"]
    
    # Second term: -β∑_(k=1)^N∑_(j=1)^C n_kj(ln(n_kj)-ξ_kj)
    # Safe log calculation to avoid log(0)
    safe_eta = np.maximum(eta, 1e-10)
    log_eta = np.log(safe_eta)
    f2 = params["beta"] * np.sum(eta * (log_eta - xi))
    
    # Third term: -σ∑_(l=0)^L w_l(ln(w_l))
    # Safe log calculation
    safe_w = np.maximum(w, 1e-10)
    log_w = np.log(safe_w)
    f3 = params["sigma"] * np.sum(w * log_w)
    
    # Fourth term: θ∑_(l=0)^L∑_(j=1)^C∑_(k=1)^N∑_(i=1)^N S_lki^S(μ_kj-μ_ij)^2
    f4 = 0
    for l in range(L + 1):
        for j in range(c):
            # Compute difference matrix: (μ_kj - μ_ij)^2 for all k,i
            u_diff = u[j].reshape(-1, 1) - u[j].reshape(1, -1)  # Shape: (N, N)
            u_diff_squared = u_diff ** 2  # Shape: (N, N)
            
            # Multiply by similarity matrix and sum
            f4 += np.sum(similarity[l] * u_diff_squared)
    f4 *= params["theta"]
    
    # Fifth term: ω∑_(l=0)^L w_l ∑_(j=1)^C∑_(k=1)^(N_L)(μ_kj-f_kj)^2 d_lkj
    f5 = 0
    if is_labeled is not None and true_clusters is not None:
        # Create mask for labeled data
        labeled_mask = is_labeled & (true_clusters >= 0)
        
        if np.any(labeled_mask):
            for l in range(L + 1):
                term = 0
                for j in range(c):
                    # Create f_kj array (1 for true class, 0 otherwise)
                    f_kj = np.zeros(N)
                    for k in np.where(labeled_mask)[0]:
                        if j == true_clusters[k]:
                            f_kj[k] = 1.0
                    
                    # Calculate squared difference for labeled points
                    diff_squared = (u[j][labeled_mask] - f_kj[labeled_mask]) ** 2
                    
                    # Multiply by distances and sum
                    term += np.sum(diff_squared * d_s[l][j][labeled_mask])
                
                f5 += w[l] * term
    f5 *= params["omega"]
    
    # Sixth term: ρ∑_(k=1)^N∑_(j=1)^C μ_kj^2
    # Vectorized sum of all squared memberships
    f6 = params["rho"] * np.sum(u ** 2)
    
    # Total objective function
    fval = f1 - f2 - f3 + f4 + f5 + f6
    
    if ISDEBUG:
        print("F value:", fval)
        print("Terms:", f1, f2, f3, f4, f5, f6)
        print("Parameters:", params)
        print("Each term contribution:")
        print("α*f1:", f1)
        print("-β*f2:", -f2)
        print("-σ*f3:", -f3)
        print("θ*f4:", f4)
        print("ω*f5:", f5)
        print("ρ*f6:", f6)
    
    return fval

def pred_cluster(u, N):
    output = np.argmax(u, axis=0)
    return output + 1


def calculate_fcm(dat, c, N, u_in):
    vi = np.zeros((c,len(dat[0])))
    u = np.copy(u_in)
    u_old = np.copy(u)
    for step in range(MAXSTEPS):
        # calculate V:
        for j in range(c):
            tg = 0
            for k in range(N):
                vi[j] += dat[k] * u[j][k]
                tg += u[j][k]
            vi[j] = vi[j] / tg
        
        d_fcm = cdist(dat, vi, metric='euclidean').T #(c,N)

        for j in range(c):
            for k in range(N):
                tg = 0
                for i in range(c):
                    tg += 1/d_fcm[i][k]
                tg *= d_fcm[j][k]
                u[j][k] = 1/tg

        diff = np.sum(u - u_old)
        if diff < 0.01:
            break
        u_old = np.copy(u)
    return u

def concensus_u_fcm(u_fcm, L, N, c):
    pred = np.zeros((L,N))
    out = np.zeros((N))
    out = out - 1
    for l in range(L):
        pre = pred_cluster(u_fcm[l], N) - 1
        for k in range(N):
            pred[l][k] = pre[k]
    for k in range(N):
        count = np.zeros((c))
        for l in range(L):
            count[int(pred[l][k])] += 1
        j = np.argmax(count)
        if count[j] == L:
            out[k] = j
    return out

def split_labeled_unlabeled(N, label):
    """Split data into labeled and unlabeled sets based on label values"""
    # Find indices of labeled (1,2,3) and unlabeled (0) samples
    labeled_indices = np.where(label > -1)[0]  # Indices of samples with labels 1,2,3
    unlabeled_indices = np.where(label == -1)[0]  # Indices of samples with label 0 (unlabeled)
    
    print(f"\nData distribution:")
    print(f"- Samples with known labels (1,2,3): {len(labeled_indices)} samples ({len(labeled_indices)/len(label)*100:.1f}%)")
    print(f"- Samples without labels (0): {len(unlabeled_indices)} samples ({len(unlabeled_indices)/len(label)*100:.1f}%)")
    
    # Print distribution of known labels
    unique_labels, counts = np.unique(label[labeled_indices], return_counts=True)
    print("\nDistribution of samples with known labels:")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"Label {lbl}: {cnt} samples ({cnt/len(label)*100:.1f}%)")
    
    # Create masked labels (-1 for unlabeled)
    masked_labels = np.copy(label)
    masked_labels[unlabeled_indices] = -1
    
    return labeled_indices, unlabeled_indices, masked_labels

def init_u_with_partial_labels(c, N, label, labeled_indices):
    """Initialize membership matrix u using partial label information with deterministic values for labeled data"""
    u = np.zeros((c, N))
    eta = np.zeros((c, N))
    xi = np.zeros((c, N))
    
    # Initialize labeled samples with deterministic values
    for k in labeled_indices:
        true_cluster = label[k] - 1  # Convert to 0-based index
        for j in range(c):
            if j == true_cluster:
                u[j][k] = 1.0    # Absolute certainty for true cluster
                eta[j][k] = 0.0  # No typicality needed for true cluster
            else:
                u[j][k] = 0.0    # No membership for wrong clusters
                eta[j][k] = 1.0  # Full typicality for wrong clusters
            xi[j][k] = 0.0       # No uncertainty for labeled data
    
    # Initialize unlabeled samples uniformly
    unlabeled_indices = np.setdiff1d(np.arange(N), labeled_indices)
    for k in unlabeled_indices:
        u[:, k] = 1.0 / c       # Uniform membership
        eta[:, k] = 0.5 / c     # Uniform typicality
        xi[:, k] = 0.2          # Some initial uncertainty
    
    return u, eta, xi

def main_MPFCM(L, c, N, rep, params, similarity, label):
    """Main MPFCM algorithm
    
    Args:
        L: Number of views
        c: Number of clusters
        N: Number of samples
        rep: Representation matrices
        params: Algorithm parameters
        similarity: Similarity matrices
        label: Ground truth labels
    """
    # Split data into labeled and unlabeled sets
    labeled_indices, unlabeled_indices, masked_labels = split_labeled_unlabeled(N, label)
    print(f"\nUsing {len(labeled_indices)} labeled samples and {len(unlabeled_indices)} unlabeled samples")

    # Create boolean array for fast label checking
    is_labeled = np.zeros(N, dtype=bool)
    is_labeled[labeled_indices] = True
    
    # Create array with true cluster assignments (-1 for unlabeled data)
    true_clusters = np.full(N, -1, dtype=int)
    for k in labeled_indices:
        if label[k] > 0:  # Only consider valid labels (1,2,3)
            true_clusters[k] = label[k] - 1  # Convert to 0-based index

    # Initialize weights for views
    w = init_w(L + 1)
    
        
    # Initialize matrices
    au_matrix = np.zeros((N, L + 1, L + 1))
    u, eta, xi = init_u_with_partial_labels(c, N, label, labeled_indices)
    
    print("\nBegin MPFCM optimization")
    v_save = []
    u_save, eta_save, xi_save, w_save = np.copy(u), np.copy(eta), np.copy(xi), np.copy(w)
    Fmin = -1
    itera = 0
    fold = -1
    u_old = np.copy(u)

    print("Begin optimization loop")    
    for step in range(100):
        # Calculate V
        v = calculate_V(u, eta, xi, rep, L, c, N, params, is_labeled, true_clusters)

        # Calculate D
        d_s = calc_D(L, c, N, v, rep)

        # Calculate u 
        u = calculate_u(u, xi, eta, w, L, c, N, d_s, params, similarity, au_matrix, 
                       is_labeled, true_clusters)
                
        # Calculate eta
        eta = calculate_eta(d_s, u, eta, xi, L, c, N, params)
                
        # Calculate xi
        xi = calculate_xi(u, eta, w, L, c, N, d_s, params)

        # Normalize all matrices
        u, eta, xi = recalc_u_eta_xi(u, eta, xi, L, c, N)

        # Update weights =
        w = calculate_w(w, d_s, u, eta, xi, L, c, N, params, similarity, is_labeled, true_clusters)

        # Calculate objective function
        F = calc_F(u, eta, xi, w, L, c, N, d_s, params, similarity, au_matrix, is_labeled, true_clusters)
        
        if Fmin == -1 or (Fmin > F and ((Fmin - F) > EPSILON)):                        
            Fmin = F
            v_save = [np.copy(v_l) for v_l in v]
            u_save, eta_save, xi_save, w_save = np.copy(u), np.copy(eta), np.copy(xi), np.copy(w)            
            itera = 0
            if ISDEBUG:
                print(f"Updated Fmin: {Fmin:.6f} at step {step}")
            
        if itera >= 5:
            break        
        itera += 1 
        fold = F            
        u_old = np.copy(u)   

    # Rest of the evaluation code remains the same
    print("\nFinal Results:")
    cluster_predict = pred_cluster(u_save, N)
    
    # Only evaluate labeled data if we have labeled indices
    metrics_labeled = None
    if len(labeled_indices) > 0:
        print("\nResults on labeled data:")
        metrics_labeled = validity(0, label[labeled_indices], cluster_predict[labeled_indices], 
                                'results/tests/pso_labeled_validity.txt', is_training=True)
    
    # Only evaluate unlabeled data if we have unlabeled indices AND they have valid labels (for research purposes)
    metrics_unlabeled = None
    if len(unlabeled_indices) > 0 and np.any(label[unlabeled_indices] > 0):
        try:
            print("\nResults on unlabeled data:")
            metrics_unlabeled = validity(0, label[unlabeled_indices], cluster_predict[unlabeled_indices], 
                                    'results/tests/pso_unlabeled_validity.txt', is_training=False)
        except Exception as e:
            print(f"Warning: Could not calculate metrics on unlabeled data: {e}")
            metrics_unlabeled = None
    else:
        print("\nSkipping unlabeled data evaluation (no ground truth available)")
    
    printValue(u_save, c, N, "results/tests/u_save.csv")

    print("\nResults with adjusted membership matrix u*(1-xi):")
    uout = u_save * (1 - xi_save)
    cluster_predict_adj = pred_cluster(uout, N)
    
    # Only evaluate labeled data with adjusted membership if we have labeled indices
    metrics_adj_labeled = None
    if len(labeled_indices) > 0:
        print("\nAdjusted results on labeled data:")
        metrics_adj_labeled = validity(0, label[labeled_indices], cluster_predict_adj[labeled_indices], 
                                    'results/tests/pso_adj_labeled_validity.txt', is_training=True)
    
    # Only evaluate unlabeled data with adjusted membership if we have unlabeled indices AND they have valid labels
    metrics_adj_unlabeled = None
    if len(unlabeled_indices) > 0 and np.any(label[unlabeled_indices] > 0):
        try:
            print("\nAdjusted results on unlabeled data:")
            metrics_adj_unlabeled = validity(0, label[unlabeled_indices], cluster_predict_adj[unlabeled_indices], 
                                        'results/tests/pso_adj_unlabeled_validity.txt', is_training=False)
        except Exception as e:
            print(f"Warning: Could not calculate adjusted metrics on unlabeled data: {e}")
            metrics_adj_unlabeled = None
    else:
        print("\nSkipping adjusted unlabeled data evaluation (no ground truth available)")
    
    printValue(uout, c, N, "results/tests/u_save_xi.csv")
    
    # Use labeled metrics if available, otherwise use unlabeled metrics
    if metrics_labeled is not None:
        return v_save, u_save, eta_save, xi_save, w_save, Fmin, metrics_labeled
    elif metrics_unlabeled is not None:
        return v_save, u_save, eta_save, xi_save, w_save, Fmin, metrics_unlabeled
    else:
        # Return empty metrics if none are available
        return v_save, u_save, eta_save, xi_save, w_save, Fmin, {}

def printValueL(val, L, c, N, filename):
    """Print values for each view to a file"""
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    f = open(filename, "w")
    for l in range(L):
        f.write("l=" + str(l) + "\n")
        for i in range(N):
            out = ''
            for j in range(c):
                out += f"{val[l][j][i]:.10f}," 
            f.write(out + "\n")
    f.close()

def printValue(val, c, N, filename):
    """Print values to a file"""
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    f = open(filename, "w")
    for i in range(N):
        out = ''
        for j in range(c):
            out += f"{val[j][i]:.10f}," 
        f.write(out + "\n")
    f.close()

def printRepSimi(rep, simi, L, c, N, filename):
    """Print representation and similarity matrices to a file"""
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    f = open(filename, "w")
    f.write("Similarity:\n")
    for l in range(L):
        f.write("view:" + str(l) + "\n")
        for i in range(N):
            out = ''
            for j in range(N):
                out += f"{simi[l][j][i]:.5f}," 
            f.write(out + "\n")

    f.write("\nRepresentation:\n")
    for l in range(L):
        f.write("rep:" + str(l) + "\n")
        for i in range(N):
            out = ''
            for j in range(c):
                out += f"{rep[l][i][j]:.5f}," 
            f.write(out + "\n")
    f.close()

def create_semi_supervised_labels(labels, labeled_ratio=0.05, random_seed=42):
    """
    Create a semi-supervised setting by masking labels for training,
    but preserving original labels for evaluation.
    
    For each class/cluster, we keep labeled_ratio% of points with labels,
    and set the rest to -1 (unlabeled) for semi-supervised learning.
    
    Args:
        labels: Original ground truth labels
        labeled_ratio: Percentage of points to keep labels for training in each class (default: 5%)
        random_seed: Random seed for reproducibility
        
    Returns:
        training_labels: Labels for training (most set to -1 for semi-supervised learning)
        evaluation_labels: Original labels for evaluation
    """
    import numpy as np
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create a copy of the original labels for training
    training_labels = np.copy(labels)
    
    # Get points with ground truth (not -1)
    valid_indices = np.where(labels != -1)[0]
    
    # Count total valid points
    total_valid = len(valid_indices)
    
    # Get unique classes (excluding -1)
    unique_classes = np.unique(labels[valid_indices])
    
    # Print initial statistics
    print(f"\nSemi-supervised learning setup:")
    print(f"- Total points: {len(labels)}")
    print(f"- Points with ground truth: {total_valid}")
    print(f"- Number of classes: {len(unique_classes)}")
    print(f"- Using {labeled_ratio*100:.1f}% of labeled points for training")
    
    # Keep track of labeled and unlabeled counts
    total_kept_labeled = 0
    total_set_unlabeled = 0
    
    # Process each class separately
    for cls in unique_classes:
        # Get indices of points belonging to this class
        class_indices = np.where(labels == cls)[0]
        class_count = len(class_indices)
        
        # Calculate how many points to keep labeled for this class
        n_keep = max(1, int(labeled_ratio * class_count))
        
        # Randomly select points to keep labeled
        keep_indices = np.random.choice(class_indices, size=n_keep, replace=False)
        
        # Set all other points of this class to unlabeled (-1)
        set_unlabeled = np.setdiff1d(class_indices, keep_indices)
        training_labels[set_unlabeled] = -1
        
        # Update counts
        total_kept_labeled += n_keep
        total_set_unlabeled += len(set_unlabeled)
        
        print(f"  Class {cls}: {class_count} points, kept {n_keep} labeled ({n_keep/class_count*100:.1f}%), set {len(set_unlabeled)} to unlabeled")
    
    # Print summary
    print("\nSummary:")
    print(f"- Total kept labeled for training: {total_kept_labeled} ({total_kept_labeled/total_valid*100:.1f}% of valid points)")
    print(f"- Total set to unlabeled for training: {total_set_unlabeled}")
    print(f"- All {total_valid} labeled points will still be used for evaluation")
    
    # Keep original labels intact for evaluation
    evaluation_labels = np.copy(labels)
    
    return training_labels, evaluation_labels

def run_MPFCM(L, c, N, rep, params, similarity, label, semi_supervised_ratio=None, simi_rep_file="results/tests/simi_rep.csv"):
    """Run the MPFCM algorithm and save results
    
    Args:
        L: Number of views
        c: Number of clusters
        N: Number of samples
        rep: Representation matrices
        params: Algorithm parameters
        similarity: Similarity matrices
        label: Ground truth labels
        semi_supervised_ratio: If not None, use this ratio for semi-supervised learning
        simi_rep_file: Output file path for similarity and representation matrices
        
    Returns:
        runtime: Execution time
        model_cluster: Cluster assignments from u
        model_cluster_adj: Cluster assignments from u*(1-xi)
    """
    start = time.time()
    
    # Create semi-supervised version of labels if requested
    if semi_supervised_ratio is not None:
        training_labels, evaluation_labels = create_semi_supervised_labels(label, labeled_ratio=semi_supervised_ratio)
        # Use training_labels for clustering
        v, u, eta, xi, w, Fmin, metrics = main_MPFCM(L, c, N, rep, params, similarity, training_labels)
        
        # Get cluster assignments from u directly
        model_cluster = pred_cluster(u, N)
        
        # Get cluster assignments from u*(1-xi)
        u_adj = u * (1 - xi)
        model_cluster_adj = pred_cluster(u_adj, N)
        
        # Calculate metrics on the complete dataset using evaluation_labels
        print("\nEvaluating clustering results on the complete dataset:")
        valid_indices = np.where(evaluation_labels != -1)[0]
        if len(valid_indices) > 0:
            print("\nMetrics for u-based clustering:")
            metrics_u = validity(0, evaluation_labels[valid_indices], model_cluster[valid_indices], 
                               'results/tests/final_evaluation_u.txt', is_training=False)
                               
            print("\nMetrics for u*(1-xi)-based clustering:")
            metrics_adj = validity(0, evaluation_labels[valid_indices], model_cluster_adj[valid_indices], 
                                'results/tests/final_evaluation_adj.txt', is_training=False)
                                
            final_metrics = metrics_adj  # Use adjusted metrics (u*(1-xi)) as requested by the user
        else:
            final_metrics = {}
            print("No ground truth labels available for evaluation.")
    else:
        v, u, eta, xi, w, Fmin, metrics = main_MPFCM(L, c, N, rep, params, similarity, label)
        
        # Get cluster assignments from u directly
        model_cluster = pred_cluster(u, N)
        
        # Get cluster assignments from u*(1-xi)
        u_adj = u * (1 - xi)
        model_cluster_adj = pred_cluster(u_adj, N)
        
        final_metrics = metrics
    
    end = time.time()
    runtime = end - start

    if ISDEBUG:
        print("\nDebug Information:")
        print("Membership matrix U:")
        printValue(u, c, N, "results/tests/u.csv")
        print("Typicality matrix Eta:")
        printValue(eta, c, N, "results/tests/eta.csv")
        print("Uncertainty matrix Xi:")
        printValue(xi, c, N, "results/tests/xi.csv")
        print("View weights W:", w)
        
        # print("\nSimilarity and Representation matrices:")
        # printRepSimi(rep, similarity, L, c, N, simi_rep_file)

    print("\nFinal cluster assignments from u:", model_cluster)
    print("\nFinal cluster assignments from u*(1-xi):", model_cluster_adj)
    
    # Return both types of cluster assignments
    return runtime, model_cluster, model_cluster_adj