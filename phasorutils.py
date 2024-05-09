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

def relu(x):
    if np.isin(type(x), [list, np.ndarray]):
        if np.isin(type(x[0]), [list, np.ndarray]):
            y = []
            for xx in x:
                y.append(relu(xx))
            y = np.array(y)
        else:
            y = np.array([max(xx,0) for xx in x])
    else:
        y = max(x, 0)
    return y

def random_unitary(size):
    phases = (np.random.random(size=size)*2. - 1)*np.pi*0.999
    return np.exp(1.j*phases)

def make_unitary(x):
    mask = np.abs(x)>1.e-5
    y = np.zeros_like(x)
    y[mask] = x[mask] / np.abs(x[mask])
    return y

def Pearson(x, Y):
    '''
     r = Pearson(x, Y)
     x is a vector, and Y is a list of vectors
    '''
    mu_x = np.mean(x)
    dx = x - mu_x
    r = []
    for yy in Y:
        mu_y = np.mean(yy)
        dy = yy - mu_y
        rr = abs(np.sum(dx*dy))
        #print(np.linalg.norm(dx), np.linalg.norm(dy))
        norm_dx = np.linalg.norm(dx)
        norm_dy = np.linalg.norm(dy)
        if norm_dx>1.e-8 and norm_dy>1.e-8:
            rr /= (np.linalg.norm(dx) * np.linalg.norm(dy))
        r.append(rr)
    if len(r)==1:
        return r[0]
    else:
        return np.array(r)

def similarity(v, S):
    '''
     max_i, sims = similarity(v, S)

     Computes the similarity of the phases in v
     to all the complex phase vectors in S (one per row).

     v is a complex vector
     S is a list of complex-valued phase maps, one in each row.

     max_i is the index of the best match
     sims is an array of similarity values
    '''
    M = len(S)

    max_sim = -1.
    max_sim_i = -1
    mean_sim = 0.
    sims = []
    for m in range(M):
        m_sim = abs(np.sum(v*np.conj(S[m]))) / np.linalg.norm(v) / np.linalg.norm(S[m])
        #print(f'{m}: {m_sim}')
        mean_sim += m_sim
        if m_sim > max_sim:
            max_sim = m_sim
            max_sim_i = m
        sims.append(m_sim)
    #print(f'Closest pattern is {max_sim_i} with similarity {max_sim}')
    #print(f'Average similarity = {mean_sim/M}')
    return max_sim_i, np.array(sims)

def permute(v):
    '''
     u = permute(v)

     Permutes the elements of v.
     Note that ipermute( permute(v) ) returns v.
    '''
    return np.roll(v, 5)

def ipermute(u):
    '''
     v = ipermute(u)

     Permutes the elements of u.
     Note that ipermute( permute(u) ) returns u.
    '''
    return np.roll(u, -5)

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

         Creates a dataset of M target patterns, each pattern being a
         N-dimensional vector of unit-modulus complex numbers.
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
            #phases = np.random.random(size=(self.K))*2.*np.pi - np.pi
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
        #thetas = arange(0, 120, 15) #[0, 30, 60, 90]
        #scales = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

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



class UniformPhaseMap():
    def __init__(self, N=400):
        '''
         ds = SparsePhaseMap(M=10, N=400, sparsity=0.1)

         Creates a dataset containing one N-dimensional phasor vector.
         The phases are regularly spaced within [-pi, pi), as they would be
         for a DFT.

         Note: N should be a perfect square!!

         ds.S.shape is (1,N)
        '''
        self.M = 1   # number of patterns
        self.N = N   # dimension of patterns
        self.cmap = plt.get_cmap('rainbow')

        self.S = np.zeros((self.M, self.N), dtype=np.csingle)

        self.S[0, :] = np.linspace(-np.pi, np.pi, self.N)

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


class SymmetricPhaseMap(SparsePhaseMap):
    def __init__(self, M=10, N=400, sparsity=0.1):
        super().__init__(M=M, N=N, sparsity=sparsity)
        self.N2 = (self.N)//2
        K2 = (self.K+1)//2
        S = []
        for k in range(M):
            angles = np.zeros(self.N)
            mags = np.zeros(self.N)
            idxs = np.random.choice(self.N2-1, K2-1, replace=False) # choose K elements
            angles[idxs] = np.random.normal(size=K2-1)*1.
            angles[-self.N2:] = np.flip(-angles[:self.N2])
            mags[idxs] = 1.
            mags[-self.N2:] = np.flip(mags[:self.N2])
            #angles = np.sort(angles)
            angles[self.N2] = 0.
            mags[self.N2] = 1.
            if N%2==0:
                angles[self.N2-1] = 0.
                mags[self.N2-1] = 1.
            S.append(np.exp(1j*angles)*mags)
        self.S = np.array(S)
