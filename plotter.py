import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os,glob
import os.path as osp
import numpy as np

def smooth(data,smooth=1):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for i,datum in enumerate(data):
            x = np.asarray(datum)
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            data[i] = smoothed_x

    return data


def tsplot(ax, data,label,**kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est,label=label,**kw)
    ax.margins(x=0)


if __name__ == '__main__':
    cnn_reward_dir = r"C:\Users\Nurullah\Desktop\TAMU\Research\logs\ddpg\distilbert-base-uncased\run52\cnn\rewards\*.npy"
    llm_reward_dir = r"C:\Users\Nurullah\Desktop\TAMU\Research\logs\ddpg\distilbert-base-uncased\run52\llm\rewards\*.npy"
    combined_reward_dir = r"C:\Users\Nurullah\Desktop\TAMU\Research\logs\ddpg\distilbert-base-uncased\run52\combined\rewards\*.npy"

    cnn_rewards_files = glob.glob(cnn_reward_dir)
    llm_rewards_files = glob.glob(llm_reward_dir)
    combined_rewards_files = glob.glob(combined_reward_dir)

    test_reward = os.path.join(cnn_reward_dir,cnn_rewards_files[0])

    cnn_rewards = []
    llm_rewards = []
    combined_rewards = []

    for i in range(len(cnn_rewards_files)):
        cnn_rewards.append(np.load(cnn_rewards_files[i]))
        llm_rewards.append(np.load(llm_rewards_files[i]))
        combined_rewards.append(np.load(combined_rewards_files[i]))

    cnn_rewards = np.array(cnn_rewards)
    llm_rewards = np.array(llm_rewards)
    combined_rewards = np.array(combined_rewards)

    smooth_cnn = smooth(cnn_rewards,smooth=20)
    smooth_llm = smooth(llm_rewards,smooth=20)
    smooth_combined = smooth(combined_rewards,smooth=20)

    fig, ax = plt.subplots()
    tsplot(ax, smooth_cnn,label="CNN-only")
    tsplot(ax, smooth_llm,label="LLM-only")
    tsplot(ax, smooth_combined, label="Combined")

    ax.legend(loc='lower right')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")

    # plt.show()
    plt.savefig("./rewards_case2.png")

    # plot_data(cnn_rewards,smooth=20)


