import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    load and plot .npy files that contain episodic rewards
    file name starts with agent and ends with seed, with delimiter '|'
    """
    data_path = './results/'

    results = {}
    num_seeds = {}

    for file_name in os.listdir(data_path):
        if file_name[-4:] != '.npy':
            continue
        file_path = os.path.join(data_path, file_name)
        
        agent = file_name.split('|')[0]
        rewards = np.load(file_path)

        if agent not in results:
            results[agent] = rewards
            num_seeds[agent] = 1
        else:
            results[agent] += rewards
            num_seeds[agent] += 1

    for agent in results:
        results[agent] /= num_seeds[agent]
        print('agent %s, number of seeds %d'
            % (str(agent), num_seeds[agent]))

    plt.figure()
    for agent in results:
        plt.plot(results[agent], label=agent)
    plt.title('cartpole average reward versus episode number')
    plt.legend()
    plt.show()
