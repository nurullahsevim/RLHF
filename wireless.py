import numpy as np

def free_space_path_loss(fc,d):
    """
    Computes the path loss in db for N points
    :param fc: the carrier frequency of transmitted signal
    :param d: the distances to N points shape of (N,)
    :return: the path loss for N points
    """
    c = 3e8
    lamda_c =c/fc
    return 10*np.log10((4*np.pi*d/lamda_c)**2)

class Transmitter:
    def __init__(self,loc,fc,Pt,Ga):
        self.loc = loc
        self.height = loc[-1]
        self.fc = fc
        self.transmit_power = Pt
        self.antenna_gain = Ga

    def get_eirp(self):
        return self.transmit_power + self.antenna_gain


def get_rssi(transmitter,d):
    path_loss = free_space_path_loss(transmitter.fc,d)
    eirp = transmitter.get_eirp()
    return eirp - path_loss