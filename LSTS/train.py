import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from PPO import PPO
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from dfa import DFA

def make_envs(params):
    env_list_names = params['training']['envs']
    max_ep_len = params['env']['max_ep_len']                   # max timesteps in one episode
    envs = []
    for env in env_list_names:
        if params['env']['render_mode']:
            env = gym.make(env, max_steps = max_ep_len, render_mode = "human")
            env = FullyObsWrapper(env)
            envs.append(env)
        else:
            env = gym.make(env, max_steps = max_ep_len)
            env = FullyObsWrapper(env)
            envs.append(env)
    return envs
        
def make_agents(state_dim_list, action_dim_list, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std):
    ppos = []
    for instance in range(len(state_dim_list)):
        ppo_agent = PPO(state_dim_list[instance], action_dim_list[instance], lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
        ppos.append(ppo_agent)
    return ppos



################################### Training ###################################
def train(params):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_list_names = params['training']['envs']
    env_num = params['training']['num_envs']
    envs = make_envs(params)
    sampling_strategy = params['training']['strategy']
    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    checkpoint_path_list = []
    for env_name in env_list_names:
        log_dir = "AGTS_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        #### get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)
        #### create new log file for each run
        log_f_name = log_dir + '/AGTS_' + sampling_strategy + "_log_" + str(run_num) + ".csv"
        print("current logging run number for " + env_name + " : ", run_num)
        print("logging at : " + log_f_name)
        # logging file
        log_f = open(log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        #####################################################
        ################### checkpointing ###################
        run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

        directory = "AGTS_Models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory = directory + '/' + env_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        random_seed = params['ppo']['random_seed']    # set random seed if required (0 = no random seed)
        checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
        checkpoint_path_list.append(checkpoint_path)
        print("save checkpoint path : " + checkpoint_path)
        #####################################################
        torch.manual_seed(params['ppo']['random_seed'])
        # for env in envs:
        #     env.seed(params['env']['random_seed'])
        np.random.seed(params['ppo']['random_seed'])
        #####################################################


    max_ep_len = params['env']['max_ep_len']                   # max timesteps in one episode
    max_training_timesteps = params['env']['max_training_timesteps']   # break training loop if timeteps > max_training_timesteps
    print_freq = params['env']['print_freq']        # print avg reward in the interval (in num timesteps)
    log_freq = params['env']['log_freq']           # log avg reward in the interval (in num timesteps)
    save_model_freq = params['env']['save_model_freq']          # save model frequency (in num timesteps)
    action_std = params['env']['action_std']                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = params['env']['action_std_decay_rate']        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = params['env']['min_action_std']                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = params['env']['action_std_decay_freq']  # action_std decay frequency (in num timesteps)
    episodes_in_each_iter = params['training']['episodes_in_each_iter']
    ################ PPO hyperparameters ################
    update_timestep = params['ppo']['update_timestep']      # update policy every n timesteps
    has_continuous_action_space = params['env']['has_continuous_action_space']  # continuous action space; else discrete
    K_epochs = params['ppo']['K_epochs']          # update policy for K epochs in one PPO update
    eps_clip = params['ppo']['eps_clip']       # clip parameter for PPO
    gamma = params['ppo']['gamma']          # discount factor
    lr_actor = params['ppo']['lr_actor']       # learning rate for actor network
    lr_critic = params['ppo']['lr_critic']      # learning rate for critic network
    # state space dimension
    state_dim_list = []
    action_dim_list = []
    for env in envs:
        state_dim = env.observation_space.shape
        state_dim_list.append(state_dim)
        # action space dimension
        if has_continuous_action_space:
            action_dim = env.action_space.shape[0]
            action_dim_list.append(action_dim)
        else:
            action_dim = env.action_space.n-1
            action_dim_list.append(action_dim)
    ppos = make_agents(state_dim_list, action_dim_list, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    #####################################################

    print("training environment name : " + env_name)
    print("============================================================================================")

    ################# training procedure ################
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    dfa_instance = DFA()
    global_timestep = 0
    environment_total_timestep = [0 for i in range(env_num)]
    environment_total_episode = [0 for i in range(env_num)]

    is_final_task = 0
    final_task_performance_timesteps = []
    final_task_performance_reward = []
    final_task_performance_done = []
    overall_performance_done = [[] for _ in range(len(envs))]
    overall_performance_timesteps = [[] for _ in range(len(envs))]

    average_timesteps_learned_tasks = [0 for _ in range(env_num)]
    while True:
        current_task = dfa_instance.choose_task()
        env = envs[current_task]
        ppo_agent = ppos[current_task]
        episodes_in_current_iter = 0
        timesteps_in_current_iter = 0
        # printing and logging variables
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0
        is_task_leared = False
        done_arr = []
        reward_arr = []
        # training loop
        while episodes_in_current_iter <= episodes_in_each_iter:

            state, info = env.reset()
            current_ep_reward = 0
            print_avg_reward = 0

            for t in range(1, max_ep_len+1):

                # select action with policy
                state['image'] = np.swapaxes(state['image'],0,2)
                state['image'] = np.expand_dims(state['image'], axis=0)
                action = ppo_agent.select_action(state['image'])    
                state, reward, terminated, truncated, info = env.step(action)

                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                if terminated or truncated:
                    ppo_agent.buffer.is_terminals.append(True)
                else:
                    ppo_agent.buffer.is_terminals.append(False)

                current_ep_reward += reward
                global_timestep += 1
                timesteps_in_current_iter +=1

                # update PPO agent
                if timesteps_in_current_iter % update_timestep == 0:
                    ppo_agent.update()

                # if continuous action space; then decay action std of ouput action distribution
                if has_continuous_action_space and timesteps_in_current_iter % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                # printing average reward
                if timesteps_in_current_iter % print_freq == 0:

                    print("Environment: {} \t\t Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(current_task, environment_total_episode[current_task], environment_total_timestep[current_task]+timesteps_in_current_iter, np.mean(reward_arr[-50:])))

                # save model weights
                if timesteps_in_current_iter % save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path_list[current_task])
                    ppo_agent.save(checkpoint_path_list[current_task])
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if terminated and current_ep_reward > 0:
                    done_arr.append(1)
                    reward_arr.append(current_ep_reward)
                    overall_performance_done[current_task].append(1)
                    overall_performance_timesteps[current_task].append(np.sum(environment_total_timestep))
                    if current_task == env_num - 1:
                        final_task_performance_reward.append(current_ep_reward)
                        final_task_performance_done.append(1)
                        final_task_performance_timesteps.append(environment_total_timestep[current_task]+timesteps_in_current_iter)
                    break
                elif terminated or truncated:
                    done_arr.append(0)
                    reward_arr.append(current_ep_reward)
                    overall_performance_done[current_task].append(0)
                    overall_performance_timesteps[current_task].append(np.sum(environment_total_timestep))                    
                    if current_task == env_num - 1:
                        final_task_performance_reward.append(current_ep_reward)
                        final_task_performance_done.append(0)
                        final_task_performance_timesteps.append(environment_total_timestep[current_task]+timesteps_in_current_iter)
                    break                    

            # if len(reward_arr) > 100 and np.mean(reward_arr[-100:]) > 0.9:
            if len(overall_performance_done[current_task]) > 100 and np.mean(overall_performance_done[current_task][-100:]) > 0.93:
                print("saving converged model at : " + checkpoint_path_list[current_task])
                ppo_agent.save(checkpoint_path_list[current_task])
                is_final_task = dfa_instance.learned_task(current_task)
                is_task_leared =True
                break
            episodes_in_current_iter += 1
            environment_total_episode[current_task] += 1
            # print_running_episodes += 1
            log_running_reward += current_ep_reward
            log_running_episodes += 1

        environment_total_timestep[current_task] += timesteps_in_current_iter
        print("done arr mean: ", np.mean(done_arr))
        if not is_task_leared:
            dfa_instance.update_teacher(current_task, np.mean(done_arr))        
        # dfa_instance.update_teacher(current_task, np.mean(done_arr))
        if is_final_task == 1:
            break
        
    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    experiment_file_name_sunk_timesteps = 'randomseed_' + str(random_seed) + '_sunk_timesteps'
    path_to_save_sunk_timesteps = log_dir + os.sep + experiment_file_name_sunk_timesteps + '.npz'

    experiment_file_name_sunk_episodes = 'randomseed_' + str(random_seed) + '_sunk_episodes'
    path_to_save_sunk_episodes = log_dir + os.sep + experiment_file_name_sunk_episodes + '.npz'

    experiment_file_name_final_reward = 'randomseed_' + str(random_seed) + '_final_reward'
    path_to_save_final_reward = log_dir + os.sep + experiment_file_name_final_reward + '.npz'

    experiment_file_name_final_dones = 'randomseed_' + str(random_seed) + '_final_dones'
    path_to_save_final_dones = log_dir + os.sep + experiment_file_name_final_dones + '.npz'

    experiment_file_name_final_timesteps = 'randomseed_' + str(random_seed) + '_final_timesteps'
    path_to_save_final_timesteps = log_dir + os.sep + experiment_file_name_final_timesteps + '.npz'

    np.savez_compressed(path_to_save_sunk_timesteps, sunk_timesteps = environment_total_timestep)
    np.savez_compressed(path_to_save_sunk_episodes, sunk_episodes = environment_total_episode)    
    np.savez_compressed(path_to_save_final_reward, final_reward = final_task_performance_reward)
    np.savez_compressed(path_to_save_final_dones, final_dones = overall_performance_done)
    np.savez_compressed(path_to_save_final_timesteps, final_timesteps = overall_performance_timesteps)

    print("Sunk timesteps: ", environment_total_timestep)
    print("Sunk episodes: ", environment_total_episode)
    print("final episodes: ", len(final_task_performance_timesteps))



if __name__ == '__main__':

    train()