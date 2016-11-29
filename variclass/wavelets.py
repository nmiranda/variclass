import numpy as np
import scipy.linalg as la
from spectrum import dpss
import matplotlib.pyplot as plt

def qmatrix(c, J, j0, mu, mesh, K = 1):
    # Q matrix for Slepian wavelet
    fl = j0 / 2**(J + 1) / mu
    fr = (j0 + 1.) / 2**(J  + 1) / mu
    mm = (2 ** J) * c
    mat = np.repeat(mesh, mm).reshape(mm, mm) - np.repeat(mesh, mm).reshape(mm, mm).T
    slep = (np.sin(2. * np.pi * mat * fr) - np.sin(2. * np.pi * mat * fl)) / (np.pi * mat)
    np.fill_diagonal(slep, 2. * (fr - fl))
    iden = np.identity(mm) - 1./mm
    slep = np.dot(iden, np.dot(slep, iden))
    return slep

def eigenvector(c, jscale, j, mu, mesh, K = 1):
    # Largest eigenvector of Q matrix
    slep = qmatrix(c, jscale, j, mu, mesh, K = K)
    evs = la.eigh(slep)[1] # Largest eigenvector is last one (eigenvalues in ascending order)
    return evs[:, -1][::-1] # This was originally negative

def slepian(c, jscale, j, mu, mesh, K = 1):
    # Calculate Slepian wavelets directly
    evs = eigenvector(c, jscale, j, mu, mesh, K = K)
    return evs / np.sqrt(2. ** jscale * mu)

def wavelet_variance(data):
    # Calculate the Slepian wavelet variance
    times, obsx, obsxerr = data
    n = len(obsx) - 2
    times = times - times[0]
    meandelta = times[n + 1] / (n + 1)
    #  meandelta = np.median(np.diff(times))
    cons = 1
    maxscale = int(np.floor(np.log2(n + 2)))
    sLj = [2**(j+1) * cons for j in range(maxscale)]
    sMj = [n - sLj[j] + 1 for j in range(maxscale)]
    mu = meandelta
    dps = []
    lamplus = []
    # Calculate Slepian wavelets
    sw_var = np.zeros(maxscale)
    tau = np.zeros_like(sw_var)
    sw_dis = np.zeros_like(sw_var)
    wc = np.empty((maxscale, n))
    ser = np.zeros([maxscale, np.max(sMj)])
    for jscale in range(maxscale):
        try:
            a, b = dpss(sMj[jscale], 3.5, 5)
        except AssertionError:
            continue
        dps.append(a)
        lamplus.append(np.repeat(1, sMj[jscale]).dot(dps[jscale]))
        for tpt in range(sMj[jscale]):
            index = range(tpt + 1, tpt + sLj[jscale] + 1)
            wf = slepian(cons, jscale + 1, 1, mu, times[index], 1)
            try:
                wc[jscale, tpt] = np.dot(obsx[index], wf)
            except ValueError:
                print jscale, tpt, index, obsx[index], wf
                sys.exit(-1)
        ser[jscale, 0:sMj[jscale]] = wc[jscale, 0:sMj[jscale]] ** 2.
    return ser
