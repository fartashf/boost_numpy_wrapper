import scipy.io as sio
import numpy as np
import gradientMex

A = sio.loadmat('M0.mat')
B = sio.loadmat('M.mat')
C = sio.loadmat('I.mat')
D = sio.loadmat('H.mat')
E = sio.loadmat('G2.mat')
F = sio.loadmat('A.mat')

print "gradientHist"
M = np.array(A['M'], order='F')
O = np.array(C['O'], order='F')
H = gradientMex.gradientHist(M, O, 2, 6, 0)
print ((H-np.array(D['H1'], order='F'))**2).sum()

print "gradient2"
I = np.array(C['I'], order='F')
Gx, Gy = gradientMex.gradient2(I)
print ((Gx-E['Gx'])**2).sum(), ((Gy-E['Gy'])**2).sum()

print "gradientMag"
I = np.array(C['I'], order='F')
c = int(C['channel'])
full = int(C['full'])
M, O = gradientMex.gradientMag(I, c, full)
print ((M-A['M'])**2).sum()

print "gradientMagNorm"
M = np.array(A['M'], order='F')
S = np.array(C['S'], order='F')
normConst = float(C['normConst'])
gradientMex.gradientMagNorm(M, S, normConst)
print ((M-B['M'])**2).sum()

print "gradientMag 2"
I = np.array(F['I'], order='F')
M, O = gradientMex.gradientMag(I, 0, 0)
