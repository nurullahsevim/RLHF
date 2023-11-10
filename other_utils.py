import numpy as np
import matplotlib.pyplot as plt

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

def get_rssi(transmitter,d):
    path_loss = free_space_path_loss(transmitter.fc,d)
    eirp = transmitter.get_eirp()
    return eirp - path_loss

def get_distance(transmitter,receivers):
    distances = np.zeros(len(receivers))
    for i in range(len(receivers)):
        distances[i] =np.linalg.norm(transmitter.loc-receivers[i].loc)
    return distances

if __name__ == '__main__':
    tr_loc = np.random.uniform(low=-500,high=500,size=(3,))
    tr_loc[-1] = 50
    N = 16
    receivers = []
    for i in range(N):
        rc_loc = np.random.uniform(low=-500,high=500,size=(3,))
        rc_loc[-1] = 0
        receivers.append(Receiver(rc_loc))
    transmitter = Transmitter(tr_loc,6e9,78.5,47.5)
    distances = get_distance(transmitter,receivers)
    rssi = get_rssi(transmitter,distances)
    fig = plt.figure()
    for i in range(N):
        plt.scatter(receivers[i].loc[0],receivers[i].loc[1],marker="x")
    plt.scatter(transmitter.loc[0],transmitter.loc[1],marker="D",s=100)
    plt.show()
