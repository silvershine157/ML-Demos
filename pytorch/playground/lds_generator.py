import numpy as np
from scipy.stats import ortho_group



### Autonomous Linear Dynamic System ###

class AutoLDS():

    def __init__(self, units):
        self.units = self.units
        self.system = np.identity(self.full) # system parameters


    # Helpers #
    def randomize_system(self):
        # better be length-preserving
        self.system = ortho_group.rvs(self.units)

    # [ units X length ]
    def make_sequence(self, initial_state, length, ts=0.1)
        A = self.system
        seq = np.zeros(self.units, length)
        seq[:, 0] = initial_state
        for t in range(length-1):
            seq[:, t+1] = seq[:,t] + ts*A*seq[:, t]
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
    - varying length
    - Non-autonomous (controlled) LDS
    - Nonlinear dynamics
    '''

def test():
    pass

test()
