import numpy as np
import importlib
import os
import sys

from mpfcm import main_MPFCM, pred_cluster
from config import NPAR, MAXSTEPS

def pso_mpfcm(L, c, N, rep, time, similarity, Aff, label):
    total_time = 0

    seed = int(time.time())
    rnp = np.random.default_rng(seed)

    alpha = rnp.uniform(0.1, 0.4, size=NPAR)
    beta = rnp.uniform(0.0001, 0.005, size=NPAR)
    sigma = rnp.uniform(0.05, 0.25, size=NPAR)
    theta = rnp.uniform(0.1, 1, size=NPAR)
    omega = rnp.uniform(0.1, 1, size=NPAR)
    ro = rnp.uniform(0.1, 1, size=NPAR)

    beta = np.copy(alpha / (N * L))
    sigma= np.copy(alpha/2)
    

    valpha = rnp.uniform(-0.1, 0.1, size=NPAR)
    vbeta = rnp.uniform(-0.1, 0.1, size=NPAR)
    vsigma = rnp.uniform(-0.1, 0.1, size=NPAR)
    vtheta = rnp.uniform(-0.1, 0.1, size=NPAR)
    vomega = rnp.uniform(-0.1, 0.1, size=NPAR)
    vro = rnp.uniform(-0.1, 0.1, size=NPAR)

    gbest = -1  
    gbestv = np.zeros((6)) 
    gbest_u = np.zeros((c, N))
    gbest_eta = np.zeros((c, N))
    gbest_xi = np.zeros((c, N))
    gbest_w = np.zeros((L + 1))

    pbest = np.zeros((NPAR))
    pbestv = np.zeros((NPAR,6))
    pbest_u = np.zeros((NPAR, c, N))
    pbest_eta = np.zeros((NPAR, c, N))
    pbest_xi = np.zeros((NPAR, c, N))
    pbest_w = np.zeros((NPAR, L + 1))

    for i in range(NPAR):
        pbest[i] = -1

    start = time.time()
    itera = 0

    for step in range(MAXSTEPS):
        print("step = ",(step,itera))
        for i in range(NPAR):
            print("particle:" + str(i) + " with parameters:" )            
            params = {
                "alpha" : alpha[i], 
                "beta" : beta[i], 
                "sigma" : sigma[i], 
                "theta" : theta[i], 
                "omega" : omega[i],
                "ro" : ro[i]
            }

            v_save, u_save, eta_save, xi_save, w_save, fmin, vali = main_MPFCM(L, c, N, rep, params, similarity, Aff, label)
            #update pbest
            vali = np.array(vali)
            fitness = np.sum(vali)

            if pbest[i] == -1 or pbest[i] < fitness:
                pbest[i] = fitness
                pbest_u[i] = np.copy(u_save)
                pbest_eta[i] = np.copy(eta_save)
                pbest_xi[i] = np.copy(xi_save)
                pbest_w[i] = np.copy(w_save)
                pbestv[i][0] = alpha[i]
                pbestv[i][1] = beta[i]
                pbestv[i][2] = sigma[i]
                pbestv[i][3] = theta[i]
                pbestv[i][4] = omega[i]
                pbestv[i][5] = ro[i]

            #update gbest
            if gbest == -1 or gbest < pbest[i]:                
                itera = 0
                gbest = pbest[i]
                gbest_u = np.copy(pbest_u[i])
                gbest_eta = np.copy(pbest_eta[i])
                gbest_xi = np.copy(pbest_xi[i])
                gbest_w = np.copy(pbest_w[i])
                gbestv = np.copy(pbestv[i])
                print("update Gbest = " + str(gbest))
                print(gbestv)
            
            # update velocity
            valpha[i] += rnp.uniform(0.1, 1) * (pbestv[i][0] - alpha[i]) + rnp.uniform(0.1, 1) * (gbestv[0] - alpha[i])
            vbeta[i] += rnp.uniform(0.1, 1) * (pbestv[i][1] - vbeta[i]) + rnp.uniform(0.1, 1) * (gbestv[1] - vbeta[i])
            vsigma[i] += rnp.uniform(0.1, 1) * (pbestv[i][2] - vsigma[i]) + rnp.uniform(0.1, 1) * (gbestv[2] - vsigma[i])
            vtheta[i] += rnp.uniform(0.1, 1) * (pbestv[i][3] - vtheta[i]) + rnp.uniform(0.1, 1) * (gbestv[3] - vtheta[i])
            vomega[i] += rnp.uniform(0.1, 1) * (pbestv[i][4] - vomega[i]) + rnp.uniform(0.1, 1) * (gbestv[4] - vomega[i])
            vro[i] += rnp.uniform(0.1, 1) * (pbestv[i][5] - vro[i]) + rnp.uniform(0.1, 1) * (gbestv[5] - vro[i])

        itera += 1
        if itera >= 5:
            print("Gbest not change for 5 times!")
            break        

        #update position
        alpha += valpha
        beta += vbeta
        sigma += vsigma
        beta  += vbeta
        sigma += vsigma
        theta += vtheta
        omega += vomega
        ro += vro

        # update for sum = 1
        alpha = np.absolute(alpha)
        beta = np.absolute(beta)
        sigma = np.absolute(sigma)
        theta = np.absolute(theta)
        omega = np.absolute(omega)
        ro = np.absolute(ro)

        suma = alpha + beta + sigma + theta + omega + ro
        alpha = alpha / suma
        beta = beta / suma
        sigma = sigma / suma
        theta = theta / suma
        omega = omega / suma
        ro = ro / suma
        
    end = time.time()        
    model_cluster = pred_cluster(gbest_u,N)

    print("Finish with gbest =",gbest)
    print("gbestv:",gbestv)
    print("gbest_u")
    print(gbest_u)
    print("gbest_eta")
    print(gbest_eta)
    print("gbest_xi")
    print(gbest_xi)
    print("gbest_w")
    print(gbest_w)

    return end - start, model_cluster, gbestv
