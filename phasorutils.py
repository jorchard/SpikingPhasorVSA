####################################
#
# (C) Jeff Orcahrd, 2024
# University of Waterloo, Canada
#
# phasorutils.py
#
####################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def random_unitary(size):
    phases = (np.random.random(size=size)*2. - 1)*np.pi*0.999
    return np.exp(1.j*phases)

def rot(d):
    '''
     R = rot(d)

     Creates a 2D rotation matrix for d degrees.
    '''
    theta = d/180*np.pi
    return np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])


class SparsePhaseMap():
    def __init__(self, M=10, N=400, sparsity=0.1):
        '''
         spm = SparsePhaseMap(M=10, N=400, sparsity=0.1)

         Creates a dataset of M target patterns, each pattern being
         an N-dimensional vector of unit-modulus complex numbers.
         Only about sparsity*D elements will be non-zero.

         Note: N should be a perfect square!!

         spm.S.shape is (N,M)
        '''
        self.M = M   # number of patterns
        self.N = N   # dimension of patterns
        self.sparsity = sparsity
        self.K = int(np.ceil(self.sparsity*N))
        self.cmap = plt.get_cmap('rainbow')

        self.S = np.zeros((self.M, self.N), dtype=np.csingle)

        for k in range(self.M):
            idxs = np.random.choice(self.N, self.K, replace=False)
            self.S[k,idxs] = random_unitary(size=(self.K)) #np.exp(1.j*phases)

    def draw_orig(self, idx=0, x=None):
        if x is None:
            S = self.S[idx]
        else:
            S = x
        phases = np.angle(S)
        rows = int(np.ceil(np.sqrt(self.N)))
        phases = ( np.reshape(phases, (rows, rows)) + np.pi ) / (2.*np.pi)
        img = self.cmap(phases)
        mags = np.reshape(abs(S), (rows, rows, 1))
        mags = np.clip(mags, 0, 1)
        mags = np.concatenate( (np.repeat(mags, 3, axis=2), np.ones((rows,rows,1))), axis=2)

        plt.imshow(img*mags);
        return

    def draw(self, idx=0, x=None, thresh=0.05):
        if x is None:
            S = self.S[idx].flatten()
        else:
            S = x

        ss = int(len(S)**0.5)
        ss_idx = ss**2
        im_cvec = np.zeros((ss, ss,3))
    #     im_cvec[:,:,3]=1
        c=0
        for i in range(ss):
            for j in range(ss):
                if np.abs(S[c]) > thresh:
                    im_cvec[i,j,:] = matplotlib.colors.hsv_to_rgb([(np.angle(S[c])/2/np.pi + 1) % 1, 1, 1])

                c+=1

        plt.imshow(im_cvec);
        return

    def similarities(self):
        mean_sim = 0.
        for r in range(self.M):
            v = self.S[r]
            for m in range(self.M):
                m_sim = abs(sum(v*np.conj(self.S[m])))
                print(f'({r},{m}): {m_sim}')
                mean_sim += m_sim
        print(f'Average similarity = {mean_sim/self.M**2}')



class HexSSP():
    def __init__(self, dtheta=5, nscales=100, thetas=None, scales=None):
        '''
         ds = HexSSP(dtheta=5, nscales=100, thetas=None, scales=None)

         Creates a dataset containing 2 N-dimensional phasor vectors that,
         in combination, generate hexagonal interference patterns (grid cells).

         Inputs:
          dtheta   increment in angle (should divide 120), eg. 4, 5, 10, 15, 20, etc.
          nscales  number of radial scales
          thetas   array of theta values (overrides dtheta)
          scales   array of scales (overrides nscales)
        '''
        Abase = np.array([[1,      -0.5,       -0.5],
                       [0, np.sqrt(3)/2, -np.sqrt(3)/2]])

        if thetas is not None:
            self.thetas = thetas
        else:
            self.thetas = np.arange(0, 120, dtheta)

        if scales is not None:
            self.scales = scales
        else:
            self.scales = np.linspace(0, np.pi*0.99, nscales)

        A = []
        for theta in self.thetas:
            for sc in self.scales:
                A.append(sc*rot(theta)@Abase)
        A = np.hstack(A)

        print(f'Number of phasors: {len(A)}')

        self.S = np.exp(1.j*A)

    def dimension(self):
        return self.S.shape[1]

    def draw(self):
        plt.plot(np.angle(self.S[0]), np.angle(self.S[1]), '.')
        plt.axis('equal')


# end