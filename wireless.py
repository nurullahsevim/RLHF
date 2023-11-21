import numpy as np
import matplotlib.pyplot as plt
import torch
import os,sys

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
    def __init__(self,N,mean,device):
        """
        :param N: number of reception points on the map
        """
        self.device = device
        self.n_receivers = N
        self.rec_mean = mean
        tr_loc = 1000*torch.rand(3)-500  # randomly choosing a location for transmitter
        tr_loc[-1] = 50  # the height of the transmitter will be fixed for now

        # tr_loc = np.random.uniform(low=-500, high=500, size=(3,))  #randomly choosing a location for transmitter
        # tr_loc[-1] = 50 #the height of the transmitter will be fixed for now
        self.initialize_transmitter(tr_loc, 6e9, 78.5, 47.5) # initialize one transmitter later should convert this to a list
        self.initialize_receivers()

    def initialize_receivers(self):

        self.receivers = []
        for i in range(self.n_receivers):
            rc_loc = 500 * (torch.rand(3)-0.5) + self.rec_mean  # randomly choosing a location for transmitter
            rc_loc[-1] = 0  # the height of the transmitter will be fixed for now
            # rc_loc = np.random.uniform(low=-500, high=500, size=(3,))
            # rc_loc[-1] = 0
            self.receivers.append(Receiver(rc_loc))

    def get_receiver_loc(self):
        locations = torch.zeros(self.n_receivers,3)
        for i in range(self.n_receivers):
            locations[i,:] = self.receivers[i].loc
        return locations.to(self.device)
    def initialize_transmitter(self,loc,fc=6e9,Pt=70,Ga=50):
        self.transmitter = Transmitter(loc,fc,Pt,Ga)
    def get_distance(self):
        self.distances = torch.zeros(self.n_receivers).to(self.device)
        for i in range(self.n_receivers):
            self.distances[i] = torch.norm(self.transmitter.loc.to(self.device) - self.receivers[i].loc.to(self.device))
            # self.distances[i] = np.linalg.norm(self.transmitter.loc - self.receivers[i].loc)
        self.distances
    def free_space_path_loss(self):
        """
        Computes the path loss in db for N points
        :return: the path loss for N points
        """
        c = 3e8
        lamda_c = c / self.transmitter.fc
        # return 10 * np.log10((4 * np.pi * self.distances / lamda_c) ** 2)
        fspl =10 * torch.log10((4 * torch.pi * self.distances / lamda_c) ** 2)
        return fspl.to(self.device)

    def get_rssi(self):
        self.get_distance()
        path_loss = self.free_space_path_loss()
        eirp = self.transmitter.get_eirp()
        return (eirp - path_loss).to(self.device)
    def visualize(self,dir_path,fig_num):
        rssi = self.get_rssi().to("cpu")
        fig = plt.figure()
        locs = self.get_receiver_loc().to("cpu").detach().numpy()
        mean_loc = np.mean(locs, axis=0)
        for i in range(self.n_receivers):
            plt.scatter(self.receivers[i].loc[0].detach().numpy(), self.receivers[i].loc[1].detach().numpy(), marker="o")
            plt.annotate("{:.2f}".format(rssi[i].detach().numpy()),self.receivers[i].loc[:2].detach().numpy(),(self.receivers[i].loc[0].detach().numpy()-20,self.receivers[i].loc[1].detach().numpy()+15),fontsize=6)
        plt.scatter(self.transmitter.loc[0].to("cpu").detach().numpy(), self.transmitter.loc[1].to("cpu").detach().numpy(), marker="D", s=200)
        plt.scatter(mean_loc[0], mean_loc[1], marker="v", s=200 , color='r')
        plt.xlim([-500, 500])
        plt.ylim([-500, 500])
        plt.savefig(os.path.join(dir_path,f"{fig_num}.png"))
        plt.close()

    def get_prompt(self):
        locations = self.get_receiver_loc()
        prompt = ""
        for i,loc in enumerate(locations):
            prompt += f"Location {i+1}: ({locations[i,0]:.2f},{locations[i,1]:.2f},{locations[i,2]:.2f}), "
        return prompt
if __name__ == '__main__':
    env = LOS_Env(16)
    print(env.get_prompt())
    env.visualize()
    # tr_loc = np.random.uniform(low=-500, high=500, size=(3,))  # randomly choosing a location for transmitter
    # tr_loc[-1] = 50  # the height of the transmitter will be fixed for now
    # env.initialize_transmitter(tr_loc,8e9,50,25)
    # env.visualize()

