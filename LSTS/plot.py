import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


# log_dir = '/homes/shukla/Work/LSTS/LSTS/AGTS_logs/MiniGrid-NineRoomsDoorGoal-v0'
log_dir = path-to-logs

random_seed = 0

experiment_file_name_sunk_timesteps = 'randomseed_' + str(random_seed) + '_sunk_timesteps'
path_to_save_sunk_timesteps = log_dir + os.sep + experiment_file_name_sunk_timesteps + '.npz'
sunk_timesteps_np = np.load(path_to_save_sunk_timesteps)['sunk_timesteps']

experiment_file_name_sunk_episodes = 'randomseed_' + str(random_seed) + '_sunk_episodes'
path_to_save_sunk_episodes = log_dir + os.sep + experiment_file_name_sunk_episodes + '.npz'
sunk_episodes_np = np.load(path_to_save_sunk_episodes)['sunk_episodes']

experiment_file_name_final_reward = 'randomseed_' + str(random_seed) + '_final_reward'
path_to_save_final_reward = log_dir + os.sep + experiment_file_name_final_reward + '.npz'
final_reward_np = np.load(path_to_save_final_reward)['final_reward']


experiment_file_name_final_dones = 'randomseed_' + str(random_seed) + '_final_dones'
path_to_save_final_dones = log_dir + os.sep + experiment_file_name_final_dones + '.npz'
final_dones_np = np.load(path_to_save_final_dones, allow_pickle=True)['final_dones']

experiment_file_name_final_timesteps = 'randomseed_' + str(random_seed) + '_final_timesteps'
path_to_save_final_timesteps = log_dir + os.sep + experiment_file_name_final_timesteps + '.npz'
final_timesteps_np = np.load(path_to_save_final_timesteps,  allow_pickle=True)['final_timesteps']

plt.figure()
final_dones_updated = [[] for i in range(final_dones_np.shape[0])]

for i in range(final_dones_np.shape[0]):
    for j in range(len(final_dones_np[i])):
        if j > 15:
            final_dones_updated[i].append(np.mean(final_dones_np[i][j-10:j-1]))
            print("i: {}; j: {}; val:{}".format(i,j,np.mean(final_dones_np[i][j-10:j-1])))
        else:
            final_dones_updated[i].append(final_dones_np[i][j])

# Plot each pair of elements from lists A and B
for i in range(final_dones_np.shape[0]):
    plt.plot(final_timesteps_np[i], final_dones_updated[i], label=f'Task{i}')

# Add labels and title
plt.xlabel('Timesteps')
plt.ylabel('Success Rate')
plt.title('Plot of Success Rate vs Timesteps')

# Add legend
plt.legend()

# Show the plot
plt.show()

print('here')