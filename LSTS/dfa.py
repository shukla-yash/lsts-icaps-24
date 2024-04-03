import numpy as np
import copy

class DFA:
    def __init__(self, strategy = "q-value"):
        self.num_envs = 5
        self.strategy = strategy
        # self.envs = ["MiniGrid-NineRoomsEasyKey-v0", "MiniGrid-NineRoomsHardKey-v0", "MiniGrid-NineRoomsKeyEasyDoor-v0", "MiniGrid-NineRoomsKeyHardDoor-v0", "MiniGrid-NineRoomsDoorGoal-v0"]
        # self.active_tasks = [0,1]
        # self.edges = {0:[0,1],1:[0,2],2:[1,3],3:[2,3],4:[3,4]} #Edge:{source node, dst node}

        self.envs = ["MiniGrid-NineRoomsHardKey-v0", "MiniGrid-NineRoomsEasyKey-v0", "MiniGrid-NineRoomsKeyHardDoor-v0", "MiniGrid-NineRoomsKeyEasyDoor-v0", "MiniGrid-NineRoomsDoorGoal-v0"]
        self.active_tasks = [0,1]
        self.edges = {0:[0,1],1:[0,2],2:[1,3],3:[2,3],4:[3,4]} #Edge:{source node, dst node}


        self.learned_tasks = []
        self.discarded_tasks = []
        self.goal_task = 4
        self.student_rewards = {enum : [] for enum in range(self.num_envs)}
        if self.strategy == "q-value":
            self.qvalue = QValue(self.num_envs, self.active_tasks)
        elif self.strategy == "ucb":
            self.qvalue = UCB(self.num_envs, self.active_tasks)
    def learned_task(self, task):
        if task == self.goal_task:
            # print("Learned goal task")
            print("Learned tasks:",self.learned_tasks)
            print("Discarded tasks:",self.discarded_tasks)            
            return 1
        # print("edges: ", self.edges)
        # print("task  to remove: ", task)
        self.active_tasks.remove(task)
        self.qvalue.teacher_q_values[task] = -np.inf
        self.learned_tasks.append(task)
        src, dst = self.edges[task][0], self.edges[task][1]
        active_task_dst = dst
        # print("Learned task:", task)
        self.edges.pop(task) 
        # print("edges after pop:", self.edges)
        nodes_to_check = []
        while True:
            edges_to_discard = []
            # print("active tasks: ", self.active_tasks)
            for key,value in self.edges.items():
                if value[1] == dst:
                    edges_to_discard.append(key)
                    self.discarded_tasks.append(key)
                    if value[0] not in nodes_to_check:
                        nodes_to_check.append(value[0])
            for item in edges_to_discard:
                self.edges.pop(item)
                self.qvalue.teacher_q_values[item] = -np.inf
                if item in self.active_tasks:
                    self.active_tasks.remove(item)
                # print("edges after discard:", self.edges)
            if len(nodes_to_check) == 0:
                break
            else:
                while True:
                    empty_nodes_to_check = 0
                    should_not_be_discarded = 0
                    if len(nodes_to_check) > 0:
                        dst = nodes_to_check.pop()
                    else:
                        empty_nodes_to_check = 1
                        break
                    for key,value in self.edges.items():
                        if dst == value[0]:
                            should_not_be_discarded = 1
                            break
                    if should_not_be_discarded == 1:
                        continue
                    else:
                        break
                if empty_nodes_to_check == 1:
                    break
                # continue
            # if len(nodes_to_check) == 0:
            #     break

        for key,value in self.edges.items():
            if value[0] == active_task_dst:
                self.active_tasks.append(key)
                self.qvalue.teacher_q_values[key] = 0
                # print("active tasks 2: ", self.active_tasks)

        return 0
    def update_teacher(self, env_num, reward):
        if len(self.student_rewards[env_num]) > 0:
            old_reward = self.student_rewards[env_num][-1]
        else:
            old_reward = 0
        self.student_rewards[env_num].append(reward)
        if self.strategy == "q-value":
            print("Current reward: {}, Prev reward : {}".format(reward, old_reward))
            reward = reward - old_reward
            self.qvalue.update_teacher_q_table(env_num,reward)
    def choose_task(self):
        if self.strategy == "q-value":
            task = self.qvalue.choose_task(self.active_tasks)
            print("Q value:" , self.qvalue.teacher_q_values)
        return task



class QValue:
    def __init__(self, num_envs, active_tasks, teacher_learning_rate = 0.1, exploration = 0.5):
        self.num_envs = num_envs
        self.active_tasks = active_tasks
        self.exploration = exploration
        self.teacher_q_values = []
        for i in range(num_envs):
            self.teacher_q_values.append(-np.inf)
        for i in active_tasks:
            self.teacher_q_values[i] = 0
        self.teacher_learning_rate = teacher_learning_rate
    def update_teacher_q_table(self, env_num, teacher_reward):
        self.teacher_q_values[env_num] = self.teacher_learning_rate*teacher_reward + (1-self.teacher_learning_rate)*self.teacher_q_values[env_num]
    def choose_task(self, active_tasks):
        if np.random.uniform() < self.exploration:
            task_number = np.random.choice(active_tasks)
        else:
            maxIndices = [i for i in range(len(self.teacher_q_values)) if self.teacher_q_values[i] == np.asarray(self.teacher_q_values).max()]
            task_number = np.random.choice(maxIndices)
        if task_number not in active_tasks:
            print("task number {} not in active tasks {}".format(task_number,active_tasks))
            print("q values {}".format(self.teacher_q_values))
            print(stop) 
        return task_number
    
class UCB:
    def __init__(self, num_envs, active_tasks, ucb_confidence_rate = 1.4, exploration =0.3):
        self.num_envs = num_envs
        self.active_tasks = active_tasks
        self.exploration = exploration
        self.teacher_q_values = []
        for i in range(num_envs):
            self.teacher_q_values.append(-np.inf)
        for i in active_tasks:
            self.teacher_q_values[i] = 0
        self.ucb_confidence_rate = ucb_confidence_rate
        self.total_times_arms_pulled = 0
        self.each_arm_count = [0 for i in range(num_envs)]
    def update_teacher_q_table(self, env_num, teacher_reward):
        self.teacher_q_values[env_num] = self.ucb_confidence_rate*teacher_reward + (1-self.ucb_confidence_rate)*self.teacher_q_values[env_num]
    def choose_task(self, active_tasks):
        self.total_times_arms_pulled += 1 
        bonus = [0 for i in range(self.num_envs)]
        ucb_values = copy.deepcopy(self.teacher_q_values)
        for i in range(self.num_envs):
            bonus[i] += self.ucb_confidence_rate*np.sqrt(np.log(self.total_times_arms_pulled)/self.each_arm_count[i]+1)
            ucb_values[i] += bonus[i]        
        task_number = np.argmax(ucb_values)
        if task_number not in active_tasks:
            print("task number {} not in active tasks {}".format(task_number,active_tasks))
            print("q values {}".format(self.teacher_q_values))
            print(stop) 
        self.each_arm_count[task_number] += 1
        return task_number    

if __name__ == '__main__':

    dfa_test = DFA()
    val = 0
    while True:
        task = dfa_test.choose_task()
        print("task chosen: ", task)
        reward = np.random.randint(0,5)
        dfa_test.update_teacher(task,reward)
        if reward == 4:
            val = dfa_test.learned_task(task)
        print("Active tasks: {}, Learned tasks: {}, Task : {}, reward = {} ".format(dfa_test.active_tasks, dfa_test.learned_tasks, task,reward))
        if val == 1:
            break
    print("here")
