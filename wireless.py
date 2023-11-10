import numpy as np
import matplotlib.pyplot as plt

class Transmitter:
    '''
    Class for base station with various parameters
    '''
    def __init__(self,loc,fc,Pt,Ga):
        '''
        :param loc: the location of the base station
        :param fc: the carrier frequency that the base station uses
        :param Pt: the transmit power of the base station
        :param Ga: the antenna gain
        '''
        self.loc = loc
        self.height = loc[-1]
        self.fc = fc
        self.transmit_power = Pt
        self.antenna_gain = Ga

    def get_eirp(self):
        """
        Returns the Effective Isotropic Radiated Power (EIRP)
        :return: EIRP
        """
        return self.transmit_power + self.antenna_gain

class Receiver:
    """
    Class for receiver points
    """
    def __init__(self,loc):
        """
        :param loc: the location where the measurement is taken
        """
        self.loc = loc

class LOS_Env:
    """
    Class for LOS environment where there is no building and all reception is line of sight
    """
    def __init__(self,N):
        """
        :param N: number of reception points on the map
        """
        self.n_receivers = N
        tr_loc = np.random.uniform(low=-500, high=500, size=(3,))  #randomly choosing a location for transmitter
        tr_loc[-1] = 50 #the height of the transmitter will be fixed for now
        self.initialize_transmitter(tr_loc, 6e9, 78.5, 47.5) # initialize one transmitter later should convert this to a list
        self.initialize_receivers()

    def initialize_receivers(self):
        
        self.receivers = []
        for i in range(self.n_receivers):
            rc_loc = np.random.uniform(low=-500, high=500, size=(3,))
            rc_loc[-1] = 0
            self.receivers.append(Receiver(rc_loc))
    def initialize_transmitter(self,loc,fc,Pt,Ga):
        self.transmitter = Transmitter(loc,fc,Pt,Ga)
    def get_distance(self):
        self.distances = np.zeros(self.n_receivers)
        for i in range(self.n_receivers):
            self.distances[i] = np.linalg.norm(self.transmitter.loc - self.receivers[i].loc)

    def free_space_path_loss(self):
        """
        Computes the path loss in db for N points
        :return: the path loss for N points
        """
        c = 3e8
        lamda_c = c / self.transmitter.fc
        return 10 * np.log10((4 * np.pi * self.distances / lamda_c) ** 2)

    def get_rssi(self):
        self.get_distance()
        path_loss = self.free_space_path_loss()
        eirp = self.transmitter.get_eirp()
        return eirp - path_loss
    def visualize(self):
        rssi = self.get_rssi()
        fig = plt.figure()
        for i in range(self.n_receivers):
            plt.scatter(self.receivers[i].loc[0], self.receivers[i].loc[1], marker="o")
            plt.annotate("{:.2f}".format(rssi[i]),self.receivers[i].loc[:2],(self.receivers[i].loc[0]-20,self.receivers[i].loc[1]+15),fontsize=6)
        plt.scatter(self.transmitter.loc[0], self.transmitter.loc[1], marker="D", s=200)
        plt.show()

if __name__ == '__main__':
    env = LOS_Env(16)
    env.visualize()

