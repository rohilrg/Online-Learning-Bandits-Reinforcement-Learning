import numpy as np
import matplotlib.pyplot as plt


# prob distribution 
def main():
    N_bandits = 10 #Number of arms
    bandit_probs = abs(np.random.normal(size=N_bandits)) #bandit probabilities
    N_experiments = 1
    N_episodes = 1
    epsilon = 0.1

    class Bandit:

        def __init__(self,bandit_probs):
            self.N = len(bandit_probs)
            self.prob = bandit_probs

        def get_reward(self,action):
            rand = abs(np.random.normal())
            reward = 1 if (rand < self.prob[action]) else 0
            return reward

    class Agent:

        def __init__(self,bandit,epsilon):
            self.epsilon = epsilon
            self.k = np.zeros(bandit.N,dtype=np.int)
            self.Q = np.zeros(bandit.N,dtype = np.float)

        def update_Q(self,action,reward):
            self.k[action] += 1
            self.Q[action] += (1./self.k[action]) * (reward - self.Q[action])

        def get_action(self,bandit,force_explore=False):
            rand = abs(np.random.normal())
            if (rand < self.epsilon):
                action_explore = np.random.randint(bandit.N)
                return action_explore
            else : 
                action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
                return action_greedy
    
    def experiment(agent,bandit,N_episodes):
        action_history = []
        reward_history = []

        for episode in range(N_episodes):
            action = agent.get_action(bandit)
            reward = bandit.get_reward(action)

            agent.update_Q(action,reward)

            action_history.append(action)
            reward_history.append(reward)

        return (np.array(action_history),np.array(reward_history))

    print("Starting process")
    reward_history_avg = np.zeros(N_episodes)
    action_history_sum = np.zeros((N_episodes,N_bandits))

    for i in range(N_experiments):
        bandit = Bandit(bandit_probs)
        agent = Agent(bandit,epsilon)

        (action_history,reward_history) = experiment(agent,bandit,N_episodes)

        if ((i+1) % (N_experiments / 100) == 0): 
            print("[Experiment {}/{}]".format(i + 1, N_experiments))
            print("  N_episodes = {}".format(N_episodes))
            print("  bandit choice history = {}".format(
                action_history + 1))
            print("  reward history = {}".format(
                reward_history))
            print("  average reward = {}".format(np.sum(reward_history) / len(reward_history)))
            print("")
        
        reward_history_avg += reward_history

        for j, (a) in enumerate(action_history):
            action_history_sum [j][a] +=1

    
    reward_history_avg /=np.float(N_experiments)
    print("reward history avg = {}".format(reward_history_avg))

    plt.plot(reward_history_avg)
    plt.xlabel("Episode number")
    plt.ylabel("Rewards collected".format(N_experiments))
    plt.title("Bandit reward history averaged over {} experiments (epsilon = {})".format(N_experiments, epsilon))
    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    plt.xlim([1, N_episodes])
    plt.show()



    plt.figure(figsize=(18, 12))
    for i in range(N_bandits):
        action_history_sum_plot = 100 * action_history_sum[:,i] / N_experiments
        plt.plot(list(np.array(range(len(action_history_sum_plot)))+1),
                 action_history_sum_plot,
                 linewidth=5.0,
                 label="Bandit #{}".format(i+1))
    plt.title("Bandit action history averaged over {} experiments (epsilon = {})".format(N_experiments, epsilon), fontsize=26)
    plt.xlabel("Episode Number", fontsize=26)
    plt.ylabel("Bandit Action Choices (%)", fontsize=26)
    leg = plt.legend(loc='upper left', shadow=True, fontsize=26)
    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    plt.xlim([1, N_episodes])
    plt.ylim([0, 100])
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(16.0)

    plt.show()

main()