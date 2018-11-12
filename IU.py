import numpy as np
import matplotlib.pyplot as plt

def main():

    N_bandits = 10
    bandits_probs = abs(np.random.normal(size=N_bandits))
    N_experiments = 100
    N_episodes = 100

    class Bandit:
        def __init__(self,bandits_probs):
            self.N = len(bandits_probs)
            self.probs = bandits_probs

        def get_reward(self,action):
            rand = abs(np.random.normal())
            reward = 1 if (rand > self.probs[action]) else 0
            return reward

    
    class Agent:
        def __init__(self,bandit):
            self.k = np.zeros(bandit.N,dtype=np.int)
            self.Q = np.zeros(bandit.N,dtype=np.float)

        def update_Q(self,action,reward):
            self.k[action] +=1
            self.Q[action] += (reward - self.Q[action])/self.k[action]

    def experiment(agent,bandit,N_episodes):

        action_history = []
        reward_history = np.zeros(bandit.N)
        for i in range(N_episodes):
            for j in range(N_bandits):
                reward = bandit.get_reward(j)

                agent.update_Q(j,reward)

                #action_history.append(action)
                reward_history[j] = reward

        print(reward_history.shape)
        return np.array(reward_history)

    reward_history_avg = np.zeros(N_bandits)

    for i in range(N_experiments):
        bandit = Bandit(bandits_probs)
        agent = Agent(bandit)

        reward_history = experiment(agent,bandit,N_episodes)

        reward_history_avg += reward_history

    reward_history_avg /=np.float(N_experiments)
    best_arm = np.argmax(reward_history_avg)
    print("Reward history avg ={}".format(reward_history_avg))
    print("The Best arm to choose after {} experiments is {} with a reward of {}".format(N_experiments,best_arm,reward_history_avg.max()))

    plt.plot(reward_history_avg)
    plt.show()



main()