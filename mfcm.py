import time
import sys
import numpy as np
from mylib import calculate_fcm, calculate_fcm_all, validity
from scipy.spatial.distance import cdist
from config import ALPHA_XI,EPSILON,EPSILON2,MAXSTEPS,init_u_eta_xi,init_w, printValidity, ISDEBUG
#from normalization import normalize_multiview_data3

from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, davies_bouldin_score

def calc_D(L, c, N, v_s, rep):
    d_s = np.zeros((L + 1, c, N))  
    r = len(rep[0][0])
    for i in range(L + 1):
       d_s[i] = cdist(rep[i], v_s[i], metric='euclidean').T #(c,N)
    return d_s

def calculate_A_u(u, eta, xi, w, L, c, N, d_s, alpha, sigma, theta, omega, similarity):
    tmp1 = np.zeros_like(u)
    tmp2 = np.zeros((N))

    for l in range(L + 1):
        tmp1 += w[l] * d_s[l]        

    for k in range(N):
        for l in range(L + 1):
            for i in range(N):
                tmp2[k] += similarity[l][k][i]

    a_u = (alpha + omega) * tmp1
    for i in range (c):
        a_u[i] += theta * tmp2
    return a_u

def calculate_Bu(u, eta, xi, A, L):    
    return np.sum(A / A[:, None], axis=1)

def calculate_G_u(u, u_fcm, L, c, N, w, d_s, theta, omega, similarity):
    G = np.zeros((c, N))
    similarity = np.array(similarity)      
    for j in range(c):  
        for k in range(N): 
            for l in range(L + 1):  
                for i in range(N):  
                    G[j][k] += theta * u[j][i] * similarity[l][k][i]
                    G[j][k] += omega * w[l] * u_fcm[l][j][k] * d_s[l][j][k]
    return G

def calculate_u(u, u_fcm, xi, eta, w, L, c, N, d_s, alpha, beta, sigma, theta, omega, similarity):
    '''
    Parameters:
        d: ma trận khoảng cách (L x c x N)
        N: số phần tử trên mỗi view
        l: chỉ số view thứ L
        L: tổng số view
        u,eta : ma trận các view (L x c x N) 
        c: số cụm 

    Return:
        u_single: ma trận (c x N)
    '''
    
    A = calculate_A_u(u, eta, xi, w, L, c, N, d_s, alpha, sigma, theta, omega, similarity)
    B = calculate_Bu(u, eta, xi, A, L)
    G = calculate_G_u(u, u_fcm, L, c, N, w, d_s, theta, omega, similarity)
    
    u_new = (1 / B)
    u_new += (G / A - (np.sum(G/A, axis = 0)) / B)
    u_new[u_new >= 1] = 1 - EPSILON
    u_new[u_new <= 0] = EPSILON
    return u_new

def calculate_eta(d_s,u,u_fcm,eta,xi,L, c, N, alpha, beta):
    eta_new = np.zeros_like(xi)
    tg = 1/(L * c)

    for j in range(c):
        for k in range(N):
            tg1 = 0
            for i in range(c):
                tg1 += xi[i][k]
            eta_new[j][k] = tg + tg1/c - xi[j][k]

    eta_new[eta_new <= 0] = EPSILON
    return eta_new

# for calculate xi
def calculate_xi(u, eta, w, L, c, N, d_s, alpha, beta, sigma):
    return 1 - (u + eta) - (1 - (u + eta)**ALPHA_XI)**(1/ALPHA_XI)    

def recalc_u_eta_xi(u,eta,xi,L,c,N):
    u[u >= 1] = 1 - EPSILON
    u[u <= 0] = EPSILON

    eta[eta >= 1] = 1 - EPSILON
    eta[eta <= 0] = EPSILON

    xi[xi >= 1] = 1 - EPSILON
    xi[xi <= 0] = EPSILON
    
    for j in range(c):
        for k in range(N):
            su = u[j][k] + eta[j][k] + xi[j][k]
            if su > 1:
                u[j][k] = u[j][k] / su
                eta[j][k] = eta[j][k] / su
                xi[j][k] = xi[j][k] / su
    return u, eta, xi

def calculate_V(u,u_fcm,eta,xi,rep, L, c, N, alpha, omega):
    v = []  # To store V_ij for each view

    for l in range(L + 1):
        vl = np.zeros((c,len(rep[l][0])))
        for j in range(c):
            tg = 0
            for k in range(N):
                tg1 = alpha * (u[j][k])**2
                tg += tg1
                vl[j] += tg1 * rep[l][k]
                tg2 = omega * (u[j][k] - u_fcm[l][j][k])**2 * rep[l][k]
                tg += tg2
                vl[j] += tg2 * rep[l][k]
        v.append(vl)  # Append to the list
    return v

def calculate_E(u,u_fcm,eta,xi,d_s,L,c,N,alpha,beta,sigma,theta, omega):
    E = np.zeros((L + 1))
    for l in range(L + 1):
        tg1 = 0
        tg2 = 0
        for j in range(c):
            for k in range(N):
                tg1 += ((u[j][k]) ** 2) * d_s[l][j][k]
                tg2 += ((u[j][k] - u_fcm[l][j][k]) ** 2) * d_s[l][j][k]
        tg1 *= theta
        tg2 *= omega
        val = (tg1 + tg2) / sigma
        
        # print("val=",val)
        E[l] = np.exp(val)
    return E

def calculate_w(w,d_s,u,u_fcm,eta,xi,L,c, N, alpha, beta, sigma, theta, omega,similarity):
    E = calculate_E(u,u_fcm,eta,xi,d_s,L,c,N,alpha,beta,sigma,theta,omega)
    tg = np.sum(E)
    w = E / tg
    return w

def calc_F(u, u_fcm, eta, xi, w, L, c, N, d_s, alpha, beta, sigma, theta, omega, similarity):
    # calculate f1

    f1 = 0
    for l in range(L):
        tg = 0
        for j in range(c):
            for k in range(N):
                tg += (((1 - xi[j][k]) * u[j][k]) ** 2) * d_s[l][j][k]
        f1 += w[l] * tg

    # calculate f3    
    f3 = np.sum(w * (np.log(w)))

    # calculate f4
    f4 = 0    
    for j in range(c):
        for k in range(N):
            for i in range(N):
                tg = 0
                for l in range(L + 1):
                    tg += similarity[l][k][i]
                f4 += tg * ((u[j][k] - u[j][i]) ** 2)

    # calculate f5
    f5 = 0    
    for l in range(L + 1):
        tg = 0
        for j in range(c):            
            for k  in range(N):
                tg += (u[j][k] - u_fcm[l][j][k]) ** 2 * d_s[l][j][k]
        f5 += w[l] * tg
    
    fval = alpha * f1 - sigma * f3 + theta * f4 + omega * f5
    # print("U val:")
    # print(u)
    if ISDEBUG:
        print("F value:",fval)
        print(f1,f3,f4,f5)
        print("alpha beta, sigma, theta, omega")
        print(alpha, beta, sigma, theta, omega)
        print("each value:")
        print(alpha * f1,sigma * f3,theta * f4,omega * f5)
    return fval

def pred_cluster(u, N):
    output = np.argmax(u, axis=0)
    return output + 1

def mutation_u(u, c, N):
    
    seed = int(time.time())

    # Create a random generator with the seed
    rng = np.random.default_rng(seed)

    u_add = rng.uniform(-0.01, 0.01, size=(c, N))
    u += u_add
    
    for k in range(N):
        tg = 0
        for j in range(c):
            tg += u[j][k]
        for j in range(c):
            u[j][k] = u[j][k] / tg
    # Initialize u and normalize across clusters

    print("U after mutation")
    print(u)

    return u


def main_MFCM(L, c, N, rep, alpha, beta, sigma, theta, omega, similarity, label):
    '''
    Hàm trả về u và v trên tất cả các view

    Parameters:
        L, N, d, c, epsilon, maxstep: number
        total_view: dữ liệu đầu vào (L x N x d)
        opt: option to init u,eta,xi
    Return:
        total_u: ma trận u trên tất cả các view (L x c x N)
        total_v: ma trận v trên tất cả các view (L x c x d)
        predict: ma trận đưa ra số cụm sẽ thuộc về trên mỗi view (L x c x 1)
        ari: đo trên từng cụm và đo trung bình
        time: thời gian thực thi
    '''

    # Random U, eta, xi
    w = init_w(L + 1)
    # Begin Fcm
    u_fcm_all, v_fcm = calculate_fcm_all(rep,c)
    range_v = np.zeros((L))
    vf = []
    st = 0
    for l in range(L):
        st += len(rep[l][0])
        range_v[l] = st        
        if l == 0:
            sta = 0                
        else:
            sta = range_v[l - 1] 
        vi = np.zeros((c,len(rep[l][0])))               
        for j in range(c):
            for d in range(len(rep[l][0])):            
                vi[j][d] = v_fcm[j][d + st]
        vf.append(vi)

    # calculate u_fcm
    u_fcm = np.zeros((L + 1, c, N))
    for l in range(L):
        d_fcm = cdist(rep[l], vf[l], metric='euclidean').T #(c,N)
        for j in range(c):
            for k in range(N):
                tg = 0
                for i in range(c):
                    tg += 1/d_fcm[i][k]
                tg *= d_fcm[j][k]
                u_fcm[l][j][k] = 1/tg
    u_fcm[L] = np.copy(u_fcm_all)

    u, eta, xi = init_u_eta_xi(c, N, u_fcm_all)

    print("Begin MPFCM")
    v_save = []
    u_save, eta_save, xi_save, w_save = np.copy(u), np.copy(eta), np.copy(xi), np.copy(w)
    Fmin = -1
    # start = time.time()
    itera = 0
    fold = -1
    u_old = np.copy(u)

    print("Begin loop")    
    for step in range(MAXSTEPS):
        # Calculate V
        v = calculate_V(u,u_fcm,eta,xi,rep,L,c,N, alpha, omega)

        # Calculate D
        d_s = calc_D(L, c, N, v, rep)

        # calculate u
        u = calculate_u(u, u_fcm, xi, eta, w, L, c, N, d_s, alpha, beta, sigma, theta, omega, similarity)

        # if itera % 10 == 1:
        #     print("itera = ",itera)

        # calculate eta
        # eta = calculate_eta(d_s,u,u_fcm,eta,xi,L, c, N, alpha, beta)
                
        # calculate xi
        # xi = calculate_xi(u, eta, w, L, c, N, d_s, alpha, beta, sigma)
        # xi = modify_Xi(u, eta, xi, xi_save, L, c, N)

        # u, eta, xi = recalc_u_eta_xi(u,eta,xi,L,c,N)

        w = calculate_w(w,d_s,u,u_fcm,eta,xi,L,c, N, alpha, beta, sigma, theta, omega,similarity)

        # Calculate F
        F = calc_F(u, u_fcm, eta, xi, w, L, c, N, d_s, alpha, beta, sigma, theta, omega, similarity)
        
        if Fmin == -1 or Fmin > F:                        
            Fmin = F
            v_save = [np.copy(v_l) for v_l in v]

            u_save, eta_save, xi_save, w_save = np.copy(u), np.copy(eta), np.copy(xi), np.copy(w)            
            itera = 0
            # if step % 10 == 0:
            print("update Fmin:" + str(Fmin) + " with step = " + str(step))

        # sub = np.abs(u - u_old)
        # if np.max(sub) < EPSILON:
        #     print("Mutation u")
        #     u = mutation_u(u, c, N)

        # u_predict = pred_cluster(u, N)
        #printValidity(label, u_predict, 1)
        #print(u)
            
        if itera >= 10 or np.abs(fold - F) < EPSILON:
         # or np.abs(fold - F) < EPSILON:                
            break        
        itera += 1 
        fold = F            
        u_old = np.copy(u)   

    if ISDEBUG:
        print("U_FCM:")
        for l in range(L + 1):
            print("l=",l)
            cluster_predict = pred_cluster(u_fcm[l], N)
            print(cluster_predict)
            validity(0, label, cluster_predict, '../results/tests/fcm_' + str(l) + '_validity.txt')
        
        printValueL(u_fcm, L, c, N, "../results/tests/u_fcm.csv")

    validity(0, label, pred_cluster(u_save, N), '../results/tests/pso_validity.txt')
    return v_save, u_save, eta_save, xi_save, w_save, Fmin

def printValueL(val, L, c, N, filename):
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
    f = open(filename, "w")
    for i in range(N):
        out = ''
        for j in range(c):
            out += f"{val[j][i]:.10f}," 
        f.write(out + "\n")
    f.close()

def printRepSimi(rep, simi, L, c, N, filename):
    f = open(filename, "w")
    f.write("Similarity:")
    for l in range(L):
        f.write("view:" + str(l) + "\n")
        for i in range(N):
            out = ''
            for j in range(N):
                out += f"{simi[l][j][i]:.5f}," 
            f.write(out + "\n")

    f.write("Representation:")
    for l in range(L):
        f.write("rep:" + str(l) + "\n")
        for i in range(N):
            out = ''
            for j in range(c):
                out += f"{rep[l][i][j]:.5f}," 
            f.write(out + "\n")
    f.close()

def run_MFCM(L, c, N, rep, alpha, beta, sigma, theta, omega, similarity,label):
    start = time.time()
    v, u, eta, xi, w, Fmin = main_MFCM(L, c, N, rep, alpha, beta, sigma, theta, omega, similarity,label)
    end = time.time()

    if ISDEBUG:
        print("U:")
        printValue(u, c, N, "../results/tests/u.csv")
        print("Eta:")
        printValue(eta, c, N, "../results/tests/eta.csv")
        print("Xi:")
        printValue(xi, c, N, "../results/tests/xi.csv")
        print("W:")
        print(w)

        printRepSimi(rep, similarity, L, c, N, "../results/tests/simi_rep.csv")

    u_predict = u
    
    model_cluster = pred_cluster(u_predict, N)
    print("Model cluster:")
    print(model_cluster)

    return end - start, model_cluster