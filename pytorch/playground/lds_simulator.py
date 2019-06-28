import numpy as np
import matplotlib.pyplot as plt

### Autonomous Linear Dynamic System ###

class AutoLDS():

    def __init__(self, units, ts=0.01, skip=50):
        self.units = units
        self.system = np.identity(self.units) # system parameters
        self.ts = ts
        self.skip = skip

    # Helpers #

    def randomize_system(self):
        # restrict to skew-symmetric matrices
        # so that state does not explode or vanish
        G = np.random.rand(self.units, self.units)
        A = G - G.transpose([1, 0])
        self.system = A

    # [ units X length ]
    def simulate(self, initial_state, length):
        frames = length * self.skip
        seq = np.zeros((self.units, length))
        state = initial_state
        for t in range(frames):
            if(t % self.skip == 0):
                seq[:, t//self.skip] = state
            state = state + self.ts * np.matmul(self.system, state)
        return seq

    ### Data Providers ###

    # all samples have uniform length for simplicity

    # [ N X units X length ] and [ N X units X units ]
    def sequence_sys_pairs(self, N, length):
        for n in range(N):
            self.randomize_system()
            pass

    # [ N X observed X length ]
    def fixed_sys_sequences(self, N, length):
        self.randomize_system() # fixed system
        for n in range(N):
            # sample initial state
            # generate sequence
            pass

    # [ N X observed X length ]
    def varying_sys_sequences(self, N, length):
        for n in range(N):
            # sample initial state
            self.randomize_system() # varying system
            # generate sequence
            pass

    '''
    Future directions:
    - noisy observation
    - observation (emission) matrix
    - varying length
    - Non-autonomous (controlled) LDS
    - Nonlinear dynamics
    '''


# test functions

visual_path = './data/visual/'
def plot_random():
    units = 3
    seq_len = 30
    alds = AutoLDS(units)
    alds.randomize_system()
    init_state = np.random.normal(0, 1, units)
    init_state = init_state/np.linalg.norm(init_state)
    seq = alds.simulate(init_state, seq_len)
    seq = seq.transpose([1, 0])    
    plt.plot(seq)
    plt.savefig(visual_path + 'LDS_simulator_test.png')

plot_random()
