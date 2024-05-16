####################################
#
# (C) Jeff Orcahrd, 2024
# University of Waterloo, Canada
#
# spikingVSA.py
#
####################################

from phasorutils import *
from brian2 import *
from brian2.units.allunits import henry
from numpy import *
from matplotlib.pyplot import *


# Cython for use in Brian2
@implementation('cython', '''
    cdef mymax(x, y):
        if x>=y:
            return x
        else:
            return y
    ''')
@check_units(x=1, y=1, result=1)
def mymax(x, y):
    return max(x,y)

@implementation('cython', '''
    cdef mymin(x, y):
        if x<=y:
            return x
        else:
            return y
    ''')
@check_units(x=1, y=1, result=1)
def mymin(x, y):
    return min(x,y)


def complex_similarity(v, S):
    M = len(S)

    max_sim = -1000.
    max_sim_i = -1
    sims = []
    for m in range(M):
        if abs(linalg.norm(v)) < 1.e-8 or abs(linalg.norm(S[m])) < 1.e-8:
            m_sim = -1000
        else:
            m_sim = real(sum(v*conj(S[m]))) / linalg.norm(v) / linalg.norm(S[m])
        if m_sim > max_sim:
            max_sim = m_sim
            max_sim_i = m
        sims.append(m_sim)
    return max_sim_i, array(sims)


def complex_similarity_2D(v, S, samp0, samp1):
    '''
     fxy = complex_similarity_2D(v, S, samp0, samp1)
     Evaluates the similarity between the complex vector v with the
     out-product of samp0 x samp1, where S[0] and S[1] are their
     corresponding axis vectors.

     ie. fxy[j,k] = dot( v , S[0]**samp0[j] * S[1]**samp1[k] )
    '''
    # Create SSP sampling of y-axis --> Sy
    Sy = []
    for yy in samp1:
        Sy.append(S[1]**(yy))
    Sy = array(Sy)

    allfx = []
    for xx in samp0:
        x = S[0]**(xx)
        bar = v * conj(x)
        
        max_i, sims = complex_similarity(bar, Sy)
        allfx.append(sims)

    return array(allfx)


# Helper functions
def spiking_similarity(pop, S, start_time):
    '''
     sims = spiking_similarity(pop, S, start_time)

     Computes the similarity of the spike trains in the population
     pop to all the complex phase vectors in S (one per row).
     This method first converts each spike train to a phase
     map using the last spike. The phase is defined by the
     population's frequency.

     pop is a SpikingPop object
     S is a list of complex-valued phase maps, one in each row.
     start_time is the start of the desired cycle

     sims is an array of similarity values
    '''
    # Convert a spike train back to a phase vector
    v = pop.spikes2complex(start_time)
    return complex_similarity(v, S)


def spatial_similarity(pop, S, xvals, start_time=0.):
    '''
     s = spatial_similarity(pop, S, xvals, start_time=0)

     Computes the similarity to a range of spatial encodings based on
     the SSP in S.

     Inputs:
      pop         spiking population
      S           SSP
      xvals       x-values to compare similarity at
      start_time  start of the desired cycle

     Output:
      s   array of similarity values corresponding to xvals
    '''
    Sx = []
    for x in xvals:
        Sx.append(S**x)
    Sx = array(Sx)

    max_i, sims = spiking_similarity(pop, Sx, start_time=start_time)

    return max_i, sims


def phase0(v):
    return mod(angle(v)+pi, 2*pi) - pi


#==================
#
# SpikingPhasorNet
#
#==================

class SpikingPhasorNet():
    def __init__(self):
        start_scope()
        #defaultclock.dt = 0.1*ms
        self.br = Network()
        self.pops = []
        self.spmons = []
        self.stmons = []
        self.syns = []
        self.symbols = {}
        self.shift = 5

    def total_neurons(self):
        tot = 0
        for pp in self.pops:
            tot += pp.total_neurons()
        return tot

    def add_pop(self, pop):
        '''
         net.add_group(pop)

         Adds the population pop to the network.
        '''
        self.pops.append(pop)
        for bro in pop.br:
            if True: #not isinstance(bro, StateMonitor):
                self.br.add(bro)
        self.symbols.update(pop.symbols)  # include dict of symbols

    def connect(self, pre, post, W=None, delays=0, phases=None):
        '''
         net.connect(pre, post, W=None, delays=0, phases=None)

         Inputs:
          pre, post  SpikingPop objects
          W          connection-weight matrix, so that
                       post = pre @ W
                     So W_ij is the weight from i to j
          delays     scalar, or (pre.N, post.N) array of delays (seconds)
          phases     (pre.N, post.N) array of phase offsets
                     Note: overrides delays argument
        '''
        syn = DelayConnection(pre, post)
        syn.br.connect()
        if W is not None:
            syn.br.mag_x = W.flatten()
            syn.br.mag_y = zeros_like(W.flatten())
        else:
            syn.br.mag_x = 0.3
            syn.br.mag_y = 0.0
        if phases is not None:
            delays = mod(array(phases)/2./pi, 1.) / pre.freq * second
        syn.br.delay = delays
        self.br.add(syn.br)  # add the Brian2 synapse to the Brian2 network
        self.syns.append(syn)

    def permute_connect(self, pre, post):
        self.connect(pre, post, W=np.roll(np.eye(pre.N), -self.shift, axis=0))

    def ipermute_connect(self, pre, post):
        self.connect(pre, post, W=np.roll(np.eye(pre.N), self.shift, axis=0))

    def convolve(self, preA, preB, post):
        synA = PhaseSumConnection(preA, post)
        self.br.add(synA.br)
        self.syns.append(synA)
        synB = PhaseSumConnection(preB, post)
        self.br.add(synB.br)
        self.syns.append(synB)

    def bundle(self, preA, preB, post):
        synA = PhaseBundleConnection(preA, post)
        self.br.add(synA.br)
        self.syns.append(synA)
        synB = PhaseBundleConnection(preB, post)
        self.br.add(synB.br)
        self.syns.append(synB)

    def deconvolve(self, preA, preB, post):
        '''
         net.deconvolve(preA, preB, post)

         Deconvolves 2 spike trains.
         post receives preA * conj(preB)

         In terms of spike times, this is the expected behaviour:
          cycle reset at 0
          preA spikes at 0.1
          preB spikes at 0.25
          post spikes at -0.15, or T-0.15, where T is the period
        '''
        synA = PhaseDiffConnectionA(preA, post)
        synB = PhaseDiffConnectionB(preB, post)
        self.br.add(synA.br)
        self.syns.append(synA)
        self.br.add(synB.br)
        self.syns.append(synB)

    def fractional_bind(self, pre, post, v):
        syn = PhaseMultConnection(pre, post, v)
        self.br.add(syn.br)
        self.syns.append(syn)

    def integrate(self, pre, post, w=1.):
        syn = IntegratorConnection(pre, post, w=w)
        self.br.add(syn.br)
        self.syns.append(syn)

    def connect_to_lmu(self, pre, lmu):
        for k,m in enumerate(lmu.m):
            self.integrate(pre, m, lmu.Bd[k])

        for c,mpre in enumerate(lmu.m):
            for r,mpost in enumerate(lmu.m):
                self.integrate(mpre, mpost, lmu.Ad[r,c])

    def reset(self, pre, post):
        for k,p in enumerate(post.pops):
            syn = ResetConnection(pre.G, p)
            self.br.add(syn.br)
            self.syns.append(syn)

    def cleanup(self, pre, post):
        '''
         net.cleanup(pre, S)

         Adds a TPAM layer that performs a memory clean-up.
         S is an (M,N) array of M N-dimensional phasor vectors
        '''
        syn = TPAMConnection(pre.G, post.G)
        self.br.add(syn.br)

    def run(self, T, **kwargs):
        self.br.run(T, namespace=self.symbols, **kwargs)

    def spike_raster(self, offset=0, color=None, **kwargs):
        total_N = 0
        for p in self.pops:
            if color is not None:
                total_N += p.spike_raster(offset=offset+total_N, color=color)
            else:
                total_N += p.spike_raster(offset=offset+total_N, **kwargs)


#==================
#
# SpikingPop: abstract base class for populations
#
#==================
class SpikingPop():
    def __init__(self):
        self.freq = 1.
        self.period = 1.
        self.symbols = {}
        self.pops = []

    def total_neurons(self):
        return self.N

    def spike_trains(self):
        return self.spmon.spike_trains()

    def spike_raster(self, offset=0, color=None, **kwargs):
        #figure()
        sp = self.spmon.spike_trains()
        N = len(sp)  # number of neurons
        y_range = [0, N-1]
        loc = offset + linspace(0, N-1, N)
        if N==1:
            bin_radius = 0.5
        else:
            bin_radius = ( loc[1] - loc[0] ) / 2.
        for k in range(N):
            nspikes = len(sp[k])
            y = [ [loc[k]-bin_radius]*nspikes, [loc[k]+bin_radius]*nspikes ]
            blah = vstack((sp[k], sp[k]))
            if color is not None:
                plot(blah, y, color=color, **kwargs);
            else:
                plot(blah, y, **kwargs);
        xlabel('Time (s)');
        ylabel('Neuron Index');
        return self.N

    def decode_xt(self, S, x_range, t_range):
        '''
         x, t = pop.decode_xt(S, x_range, t_range)

         Decodes the x-value stored in a spiking population over time.

         Inputs:
           S        complex vector that the SSP is based on
           x_range  (tuple) lower and upper range of x values
           t_range  (tuple) start and end time of decoding

         Outputs:
           x         array of the best-fit x values
           t         array of times (the start of each cycle)
        '''
        n_samples = 2001
        xvals = linspace(x_range[0], x_range[1], n_samples)
        tstart = t_range[0]
        tstop = t_range[1]
        tvals = arange(tstart, tstop, 1./self.freq)
        fx = []
        for tt in tvals:
            max_i, sims = spatial_similarity(self, S, xvals, start_time=tt)
            fx.append(xvals[max_i])
        return array(fx), tvals


    def spikes2complex(self, start_time):
        '''
         v = sp.spikes2complex(start_time)

         Looks at one period of the spike trains and converts it to
         a complex vector where the phases match the spike times.
         Neurons that don't spike get mapped to 0.
        '''
        sp = self.spmon.spike_trains()

        v = []
        for k in range(len(sp)):
            spike_time = []
            st = sp[k]/second
            for s in st:
                if s>=start_time and s<start_time+self.period:
                    spike_time.append(s)
            #if len(spike_time)>1:
            #    print(f'Oops, found {len(spike_time)} spikes in the time [{start_time},{start_time+self.period}]')
            if len(spike_time)>0:
                delay = 2j*pi*(spike_time[0]-start_time)/self.period
                v.append(exp(delay))
            else:
                v.append(0j)
        #delay = array([mod(sp[k][-1]/second, T) for k in range(self.N)])
        #delay = array(delay)
        #v = exp(2j*pi*delay/0.2)
        return v

    def closest_pattern(self, S, start_time):
        '''
         k = closest_pattern(S)

         Finds the phase map that is the closest match to the list
         of spike trains.
         This method first converts each spike train to a phase
         map using the last spike. The phase is defined by the
         population's frequency.

         S is a list of complex-valued phase maps, one in each row.

         k is the index of the closest match, as determined by
           the cosine distance
        '''
        # Convert a spike train back to a phase vector
        v = self.spikes2complex(start_time)

        M = len(S)

        max_sim = -1.
        max_sim_i = -1
        mean_sim = 0.
        for m in range(M):
            m_sim = abs(sum(v*conj(S[m]))) / linalg.norm(v) / linalg.norm(S[m])
            print(f'{m}: {m_sim}')
            mean_sim += m_sim
            if m_sim > max_sim:
                max_sim = m_sim
                max_sim_i = m
        print(f'Closest pattern is {max_sim_i} with similarity {max_sim}')
        print(f'Average similarity = {mean_sim/M}')
        return max_sim_i


#==================
#
# Extensions of SpikingPop
#
#==================

class GenerateSpikes(SpikingPop):
    def __init__(self, N=1, indices=[1], times=[0]):
        '''
         gen = GenerateSpikes(N=1, indices=[1], times=[0])

         Creates a population of spiking neurons.
         These do not repeat. If you want periodic spikes, use
         GenerateSP instead.

         Inputs:
          N        number of neurons
          indices  indices of spiking neurons
          times    times of spikes
        '''
        super().__init__()
        self.N = N
        self.indices = indices
        self.times = times
        self.G = SpikeGeneratorGroup(self.N, self.indices, array(self.times)*second)
        self.spmon = SpikeMonitor(self.G)
        self.br = [self.G, self.spmon]


class GenerateSP(SpikingPop):
    def __init__(self, N=64, freq=5., delays=0, C=None):
        '''
         gen = GenerateSP(N=64, freq=5., delays=0, C=None)

         Creates a population of spiking phasors.

         Inputs:
          N       number of neurons
          freq    common oscillation frequency
          delays  spike delays (seconds)
          C       complex-valued states for the oscillators. If the abs
                  of a value is below 0.8, the neuron will be silent.
                  (this option overrides the delays input)
        '''
        super().__init__()
        self.N = N
        self.freq = freq
        self.period = 1./self.freq
        # Represent phases as time delays
        if C is not None:
            v = abs(C)
            phases = angle(C)
            delays = array(phases)/2./pi * self.period
            select = v>=0.8
            indices = arange(self.N)[select]
            delays = delays[select]
        else:
            indices = list(range(self.N))
            delays = zeros(self.N) + array(delays)
        self.indices = indices
        # For some reason, we have to run mod twice to avoid delay=period errors ???
        self.delays = mod(float32(delays), self.period)
        self.delays = mod(self.delays, self.period)
        # Avoid having a delay in the last tick of a period (??!)
        mask = self.delays>(1./self.freq - 0.0001)
        self.delays[mask] = 1./self.freq - 0.0001
        # Create Brian2 objects
        self.G = SpikeGeneratorGroup(N, indices, self.delays*second, period=self.period*second)
        self.spmon = SpikeMonitor(self.G)
        self.br = [self.G, self.spmon]

    def set_spikes(self, C=None):
        if C is not None:
            v = abs(C)
            phases = angle(C)
            delays = array(phases)/2./pi * self.period
            select = v>=0.8
            indices = arange(self.N)[select]
            delays = delays[select]
        else:
            indices = list(range(self.N))
            delays = zeros(self.N) + array(delays)
        self.indices = indices
        self.delays = mod(float32(delays), self.period)
        self.delays = mod(self.delays, self.period)
        # Avoid having a delay in the last tick of a period (??!)
        mask = self.delays>(1./self.freq - 0.0001)
        self.delays[mask] = 1./self.freq - 0.0001
        self.G.set_spikes(self.indices, self.delays*second, period=1./self.freq*second)
        self.pops = [self.G]


class ResetConnection():
    def __init__(self, preG, postG):
        '''
         c = ResetConnection(preG, postG)
         Note that preG and postG refer to neuron groups, not SpikingPop
         objects.
        '''
        syn_on_pre = '''
            p_post = 0.
            '''
        self.br = Synapses(preG, postG, on_pre=syn_on_pre, method='euler')
        self.br.connect()


class IntegratorConnection():
    def __init__(self, pre, post, w=1.):
        syn_eqs = 'w : 1'
        syn_on_pre = '''
            p_post -= w*x_post
            '''
        self.br = Synapses(pre.G, post.G, syn_eqs, on_pre=syn_on_pre, method='euler')
        self.br.connect(j='i')
        self.br.w = w


class SPIntegrator(SpikingPop):
    def __init__(self, N=64, freq=5.):
        '''
         gen = SPIntegrator(N=64, freq=5.)

         Creates a population of spiking phasors that integrate
         the phases of the incoming spikes.

         Inputs:
          N       number of neurons
          freq    common oscillation frequency
        '''
        super().__init__()
        self.N = N
        self.freq = freq
        self.period = 1./self.freq
        self.symbols = {'lam': self.freq}

        eqs = Equations('''
            dx/dt = lam * hertz : 1
            dp/dt = lam * hertz : 1
            ''')

        reset_eqs = '''
            p -= 1.
            '''

        wrap_reset = '''
            x = -0.5
            '''

        self.G = NeuronGroup(self.N,
                            model=eqs,
                            threshold='p>=1',
                            reset=reset_eqs,
                            refractory=0.*ms,
                            events={'wrap': 'x>=0.5'},
                            method='euler')
        self.G.run_on_event('wrap', wrap_reset)
        self.G.x = 0.
        self.G.p = 0.
        self.stmon = StateMonitor(self.G, True, record=True)
        self.spmon = SpikeMonitor(self.G)
        self.br = [self.G, self.stmon, self.spmon]
        self.pops = [self.G]


# # TPAM
# class TPAMConnection():
#     def __init__(self, pre, post):
#         self.pre = pre
#         self.post = post


#         syn_eqs = '''
#             mag_v : 1
#             mag_u : 1
#             '''
#         syn_on_pre = '''
#             Vs_post += mag_v
#             Us_post += mag_u
#             '''

#         # Input connections
#         self.br = Synapses(pre.G, post.G,
#                             syn_eqs, on_pre=syn_on_pre,
#                             method='euler')
#         self.br.connect(j='i')
#         self.br.mag_u = 0.
#         self.br.mag_v = 0.5

# class TPAMPop(SpikingPop):
#     def __init__(self, targets=None, freq=5.):
#         '''
#          net = TPAMLayer(targets, freq=5.)

#          Creates a recurrent TPAM layer for a given set of phasor targets.

#          Inputs:
#           targets  (M,N) array of target M N-dim phasor patterns
#           freq     baseline frequency
#         '''
#         self.freq = freq  # Baseline frequency
#         self.period = 1./self.freq
#         self.S = targets
#         self.M, self.N = self.S.shape
#         self.Ks = sum(abs(self.S[0,:])>1.e-5)

#         # W has phase shifts in radians
#         W = self.S.T@np.conj(self.S) / self.Ks  # ds.S = pvec.T from Paxon's code
#         self.W = W
#         for k in range(len(W)):
#             W[k,k] = 0.

#         self.symbols = {'theta': self.freq}

#         eqs = Equations('''
#             dVs/dt = (-2*pi*theta*Us - 0.4*Vs + I_ext) * hertz : 1
#             dUs/dt = ( 2*pi*theta*Vs - 0.4*Us) * hertz : 1
#             I_ext : 1
#             ''')

#         syn_eqs = '''
#             mag_v : 1
#             mag_u : 1
#             '''
#         syn_on_pre = '''
#             Vs_post += mag_v
#             Us_post += mag_u
#             '''

#         # Phasor neuron group
#         self.G = NeuronGroup(self.N,
#                              model=eqs,
#                              threshold='Vs>0.9 and Us>0',
#                              reset='Vs=0.7',
#                              refractory=0.75*1./self.freq*second,
#                              method='euler')
#         self.G.Vs = 0.3*ones(self.N)
#         self.G.Us = 0.0*ones(self.N)
#         self.G.I_ext = 0.

#         # Recurrent connections
#         self.G2G = Synapses(self.G, self.G, syn_eqs, on_pre=syn_on_pre, method='euler')
#         i_pre, j_post = meshgrid(range(self.N), range(self.N), indexing='ij')
#         self.G2G.connect(i=i_pre.flatten(), j=j_post.flatten())
#         syn_abs = abs(W.T).flatten()
#         syn_thresh = 0.1 / (2.*self.N)**0.5
#         syn_idxs = where(syn_abs <= syn_thresh)[0]  # From Frady
#         syn_abs[syn_idxs] = 0.
#         syn_phase = mod(angle(W.T).flatten(), 2.*pi)  # can only deal with positive delays
#         syn_g = 0.3
#         self.G2G.mag_v = syn_g * syn_abs
#         self.G2G.mag_u = 0. #syn_g * syn_abs * sin(syn_phase) * VU_factor
#         self.G2G.delay = syn_phase/2./pi * self.period * second

#         # Monitors
#         self.spmon = SpikeMonitor(self.G)
#         self.stmon = StateMonitor(self.G, True, record=True)
#         self.br = [self.G, self.G2G, self.spmon, self.stmon]
#         self.pops = [self.G]




class DelayConnection():
    def __init__(self, pre, post):
        syn_eqs = '''
            mag_x : 1
            mag_y : 1
            '''
        syn_on_pre = '''
            x_post += mag_x
            '''
        self.br = Synapses(pre.G, post.G, syn_eqs, on_pre=syn_on_pre, method='euler')


# Fractional Binding
class PhaseMultConnection():
    def __init__(self, pre, post, v):
        '''
         syn = PhaseMultConnection(pre, post, v)
         pre, post are SpikingPop objects, and
         v is the phase multiplier (scalar)
        '''
        self.id = random.randint(1000)
        self.alpha_id = f'alpha{self.id}'
        syn_eqs = f'{self.alpha_id} : 1'
        syn_on_pre = f'''
            th_post = ((x_post*{self.alpha_id}+0.5) % 1) - 0.5
            '''
        self.br = Synapses(pre.G, post.G, syn_eqs, on_pre=syn_on_pre, method='euler')
        self.br.connect(j='i')
        setattr(self.br, self.alpha_id, v)
        self.stmon = StateMonitor(self.br, True, record=True)

class PhaseMultPop(SpikingPop):
    def __init__(self, N=64, freq=5.):
        super().__init__()
        self.N = N
        self.freq = freq
        self.period = 1./self.freq
        self.symbols = {'mymax': mymax, 'mymin': mymin, 'tau': 1./self.freq}

        eqs = Equations('''
            dx/dt = b / tau * hertz : 1
            b : 1
            th : 1
            refr : 1
            ''')

        reset_eqs = '''
            refr = 1
            '''

        wrap_reset = '''
            x = -0.5
            refr = 0
            '''

        pwrap_down = '''
            th -= 1.
            '''

        pwrap_up = '''
            th += 1.
            '''

        go_dormant = '''
            b = 0.
            th = 1.
            '''

        self.G = NeuronGroup(self.N,
                            model=eqs,
                            threshold='x>=th and refr==0',
                            reset=reset_eqs,
                            refractory=1./self.freq/2.*ms,
                            events={'wrap': 'x>=0.5',
                                    'pdown': 'th>0.5',
                                    'pup': 'th<-0.5'},
                            method='euler')
        self.G.x = 0.
        self.G.b = 1.   # init b=1 'not dormant'
        self.G.th = 0.5
        self.G.refr = 0
        self.G.run_on_event('wrap', wrap_reset)
        self.G.run_on_event('pdown', pwrap_down)
        self.G.run_on_event('pup', pwrap_up)

        self.stmon = StateMonitor(self.G, True, record=True)
        self.spmon = SpikeMonitor(self.G)
        self.br = [self.G, self.spmon, self.stmon]
        self.pops = [self.G]


# Binding
class PhaseSumConnection():
    def __init__(self, pre, post):
        syn_eqs = '''
            w : 1
            '''
        syn_on_pre = '''
            q_post += mymax(qdot,0.) * x_post
            qdot_post -= 1
            '''
        self.br = Synapses(pre.G, post.G, syn_eqs, on_pre=syn_on_pre, method='euler')
        self.br.connect(j='i')

class PhaseSumPop(SpikingPop):
    def __init__(self, N=64, freq=5.):
        super().__init__()
        self.N = N
        self.freq = freq
        self.period = 1./self.freq
        self.symbols = {'mymax': mymax, 'mymin': mymin, 'tau': 1./self.freq}

        eqs = Equations('''
            dx/dt = ( 1. / tau ) * hertz : 1
            dq/dt = mymin(qdot, 0.) / tau * hertz : 1
            qdot : 1
            ''')

        reset_eqs = '''
            q = 0
            qdot = 1
            '''

        wrap_reset = '''
            x = 0.
            '''

        self.G = NeuronGroup(self.N,
                            model=eqs,
                            threshold='q<0',
                            reset=reset_eqs,
                            refractory=0.*1./self.freq/2.*ms,
                            events={'wrap': 'x>=1'},
                            method='euler')
        self.G.q = 0.
        self.G.qdot = 1.
        self.G.run_on_event('wrap', wrap_reset)

        self.stmon = StateMonitor(self.G, True, record=True)
        self.spmon = SpikeMonitor(self.G)
        self.br = [self.G, self.stmon, self.spmon]
        self.pops = [self.G]


# Unbinding
class PhaseDiffConnectionA():
    def __init__(self, pre, post):
        syn_eqs = '''
            w : 1
            '''
        # This should be the second spike, used to set the threshold.
        # If p_post==0 (ie. no 1st spike yet)
        #    then set threshold to 1.1 (out of reach, dormant, waiting)
        syn_on_pre = '''
            pdot_post = 0.
            Dphi_post = (1-sign(p_post))*1.1 + p_post
            '''
        self.br = Synapses(pre.G, post.G, syn_eqs, on_pre=syn_on_pre, method='euler')
        self.br.connect(j='i')
        self.br.connect(j='i')

class PhaseDiffConnectionB():
    def __init__(self, pre, post):
        syn_eqs = '''
        w : 1
        '''
        syn_on_pre = '''
            pdot_post = 1.
            p_post = 0
            '''
        self.br = Synapses(pre.G, post.G, syn_eqs, on_pre=syn_on_pre, method='euler')
        self.br.connect(j='i')
        self.br.connect(j='i')

class PhaseDiffPop(SpikingPop):
    '''
     Computes spiketime(A) - spiketime(B)
     That is, it measures the time that lapses after B spikes, until A spikes.
     The spike from A might not arrive until the next cycle.
    '''
    def __init__(self, N=64, freq=5.):
        super().__init__()
        self.N = N
        self.freq = freq
        self.period = 1./self.freq
        self.symbols = {'tau': 1./self.freq}

        eqs = Equations('''
            dx/dt = 1. / tau * hertz : 1
            dp/dt = pdot / tau * hertz : 1
            pdot : 1
            Dphi : 1
            refr : 1
            ''')

        reset_eqs = '''
            refr = 1.
            '''

        wrap_reset = '''
            x = 0.
            refr = 0.
            '''

        self.G = NeuronGroup(self.N,
                            model=eqs,
                            threshold='x>Dphi and refr==0.',
                            reset=reset_eqs,
                            refractory=1./self.freq/2.*ms,
                            events={'wrap': 'x>=1'},
                            method='euler')
        self.G.x = 0.
        self.G.p = 0.
        self.G.pdot = 0.
        self.G.Dphi = 1.1
        self.G.run_on_event('wrap', wrap_reset)

        self.stmon = StateMonitor(self.G, True, record=True)
        self.spmon = SpikeMonitor(self.G)
        self.br = [self.G, self.stmon, self.spmon]
        self.pops = [self.G]


# Hopfield Clean-Up Memory
class SpikingModernHopfield(SpikingPop):
    def __init__(self, targets, freq=5.):
        '''
         net = SpikingModernHopfield(targets, freq=5.)

         Creates a layer that implements a modern Hopfield network of spiking
         phasors. The layer consists of 2 parts: the feature layer, and the
         hidden layer.

         Inputs:
          targets  (M,N) array of target M N-dim phasor patterns
          freq     baseline frequency
        '''
        self.freq = freq  # Baseline frequency
        self.period = 1./self.freq
        self.S = targets
        self.M, self.N = self.S.shape
        self.Ks = sum(abs(self.S[0,:])>1.e-5)
        self.inhib_tau = 0.05 * second

        self.symbols = {'theta': self.freq, 'inhib_tau': self.inhib_tau}

        geqs = Equations('''
            dx/dt = (-2*pi*theta*y - 0.4*x) * hertz : 1
            dy/dt = ( 2*pi*theta*x - 0.4*y) * hertz : 1
            ''')

        heqs = Equations('''
            dx/dt = (-2*pi*theta*y - (0.02 + inhib)*x) * hertz : 1
            dy/dt = ( 2*pi*theta*x - (0.02 + inhib)*y) * hertz : 1
            dinhib/dt = -inhib / inhib_tau  : 1
            ''')

        g2h_eqs = '''
            w : 1
            '''
        g2h_on_pre = '''
            x_post += w
            '''
        h2h_eqs = '''
            w : 1
            '''
        h2h_on_pre = '''
            inhib_post += w
            '''

        # Feature Phasor neuron group
        self.G = NeuronGroup(self.N,
                             model=geqs,
                             threshold='x>0.9 and y>0',
                             reset='x=0.7',
                             refractory=0.75*1./self.freq*second,
                             method='euler')
        self.G.x = 0.3*ones(self.N)
        self.G.y = 0.0*ones(self.N)

        # Hidden Phasor neuron group
        self.H = NeuronGroup(self.M,
                             model=heqs,
                             threshold='x>0.9 and y>0',
                             reset='x=0.7',
                             refractory=0.75*1./self.freq*second,
                             method='euler')
        self.H.x = 0.3*ones(self.M)
        self.H.y = 0.0*ones(self.M)
        self.H.inhib = 0.

        # G->H connections
        self.G2H = Synapses(self.G, self.H, g2h_eqs, on_pre=g2h_on_pre, method='euler')
        i_pre, j_post = meshgrid(range(self.N), range(self.M), indexing='ij')
        self.G2H.connect(i=i_pre.flatten(), j=j_post.flatten())
        syn_abs = abs(self.S.T).flatten() / self.Ks
        syn_phase = mod(-angle(self.S.T).flatten(), 2.*pi)  # can only deal with positive delays
        syn_g = 0.6
        self.G2H.w = syn_g * syn_abs
        self.G2H.delay = syn_phase/2./pi * self.period * second

        # H->G connections
        self.H2G = Synapses(self.H, self.G, g2h_eqs, on_pre=g2h_on_pre, method='euler')
        i_pre, j_post = meshgrid(range(self.M), range(self.N), indexing='ij')
        self.H2G.connect(i=i_pre.flatten(), j=j_post.flatten())
        syn_abs = abs(self.S).flatten()
        syn_phase = mod(angle(self.S).flatten(), 2.*pi)  # can only deal with positive delays
        syn_g = 0.5
        self.H2G.w = syn_g * syn_abs
        self.H2G.delay = syn_phase/2./pi * self.period * second

        # H->H connections
        self.H2H = Synapses(self.H, self.H, h2h_eqs, on_pre=h2h_on_pre, method='euler')
        self.H2H.connect(condition='i != j')
        self.H2H.w = 3.

        # Monitors
        self.Gspmon = SpikeMonitor(self.G)
        self.Hspmon = SpikeMonitor(self.H)
        self.Gstmon = StateMonitor(self.G, True, record=True)
        self.Hstmon = StateMonitor(self.H, True, record=True)
        self.br = [self.G, self.H, self.G2H, self.H2G, self.H2H, self.Gspmon, self.Hspmon, self.Gstmon, self.Hstmon]
        self.pops = [self.G, self.H]

    def total_neurons(self):
        return self.N + self.M

    def spike_raster(self, offset=0, color=None, **kwargs):
        n_rasters = self.spike_raster_G(offset=offset, color=color)
        offset += n_rasters
        m_rasters = self.spike_raster_H(offset=offset, color=color)
        return n_rasters + m_rasters

    def spike_raster_G(self, offset=0, color=None, **kwargs):
        sp = self.Gspmon.spike_trains()
        m_rasters = self._spike_raster(sp, offset=offset, color=color)
        return m_rasters

    def spike_raster_H(self, offset=0, color=None, **kwargs):
        sp = self.Hspmon.spike_trains()
        m_rasters = self._spike_raster(sp, offset=offset, color=color)
        return m_rasters

    def _spike_raster(self, sp, offset=0, color=None, **kwargs):
        N = len(sp)  # number of neurons
        y_range = [0, N-1]
        loc = offset + linspace(0, N-1, N)
        if N==1:
            bin_radius = 0.5
        else:
            bin_radius = ( loc[1] - loc[0] ) / 1.9
        for k in range(N):
            nspikes = len(sp[k])
            y = [ [loc[k]-bin_radius]*nspikes, [loc[k]+bin_radius]*nspikes ]
            blah = vstack((sp[k], sp[k]))
            if color is not None:
                plot(blah, y, color=color, **kwargs);
            else:
                plot(blah, y, **kwargs);
        xlabel('Time (s)');
        ylabel('Neuron Index');
        return N

    def spikes2complex(self, start_time):
        '''
         v = sp.spikes2complex(start_time)

         Looks at one period of the spike trains and converts it to
         a complex vector where the phases match the spike times.
         Neurons that don't spike get mapped to 0.
        '''
        sp = self.Gspmon.spike_trains()

        v = []
        for k in range(len(sp)):
            spike_time = []
            st = sp[k]/second
            for s in st:
                if s>=start_time and s<start_time+self.period:
                    spike_time.append(s)
            if len(spike_time)>0:
                delay = 2j*pi*(spike_time[0]-start_time)/self.period
                v.append(exp(delay))
            else:
                v.append(0j)
        return v

# Simple relay
class SpikingRelayPop(SpikingPop):
    def __init__(self, N=64, start_at=0, stop_at=1000000):
        '''
         pop = SpikingRelayPop(N=64, start_at=0, stop_at=1000000)

         A population that simply spikes every time a spike arrives.
        '''
        super().__init__()
        self.N = N
        self.id = random.randint(10000)
        self.start_id = f'start{self.id}'
        self.stop_id = f'stop{self.id}'

        eqs = Equations(
            'x : 1\n'
            'd : 1\n'
            f'{self.start_id} : second\n'
            f'{self.stop_id} : second')

        wake = '''
            x = 0.
            d = 1
            '''

        sleep = '''
            d = 2
            '''

        self.G = NeuronGroup(self.N,
                            model=eqs,
                            threshold='x>0.1 and d==1',
                            reset='x=0',
                            events={'wake': f't>{self.start_id} and d==0',
                                    'sleep': f't>{self.stop_id} and d==1'},
                            method='euler')
        self.G.x = 0.
        self.G.d = 0.
        setattr(self.G, self.start_id, start_at*second)
        setattr(self.G, self.stop_id, stop_at*second)
        self.G.run_on_event('wake', wake)
        self.G.run_on_event('sleep', sleep)
        self.stmon = StateMonitor(self.G, True, record=True)
        self.spmon = SpikeMonitor(self.G)
        self.br = [self.G, self.spmon, self.stmon]
        self.pops = [self.G]



# Spiking Resonator
class SpikingPhasorPop(SpikingPop):
    def __init__(self, N=64, freq=5, stop_at=1000.):
        '''
         spop = SpikingPhasorPop(N=64, freq=5, stop_at=1000.)

         Creates a population of spiking phasors.
         The population stops spiking after the specified number
         of seconds.
        '''
        super().__init__()
        self.N = N
        self.freq = freq
        self.period = 1./self.freq
        self.symbols = {'theta': freq, 'stop': stop_at*second}

        eqs = Equations('''
            dx/dt = ( -2*pi*theta*y * I_ext - 50.*(x+2)*(1.-I_ext) )*hertz : 1
            dy/dt = (  2*pi*theta*x * I_ext - 50.*y*(1.-I_ext) )*hertz : 1
            I_ext : 1
            ''')

        reset_eqs = '''
            x = 0.8
            '''
        dormant = '''
            I_ext = 0.0
            '''

        self.G = NeuronGroup(self.N,
                            model=eqs,
                            threshold='x>1 and y>0',
                            reset=reset_eqs,
                            refractory=0*1./self.freq/2.*ms,
                            events={'go_dormant': 't>stop'},
                            method='euler')
        self.G.x = 0.8 * ones(self.N)
        self.G.y = 0.0 * ones(self.N)
        self.G.I_ext = 1.
        self.G.run_on_event('go_dormant', dormant)

        self.stmon = StateMonitor(self.G, True, record=True)
        self.spmon = SpikeMonitor(self.G)
        self.br = [self.G, self.stmon, self.spmon]
        self.pops = [self.G]

# end