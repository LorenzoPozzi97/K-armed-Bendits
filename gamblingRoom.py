import gym
import numpy as np
from gym import spaces

class slotMachine:
    """
        A slot machine contains a reward distribution that randomly generated with restricted mean and standard deviation. 
            sample function: generates a reward at each time step based on the given reward distribition
    """
    def __init__(self):
        self.mu = np.random.uniform(-5, 5)  # mean
        self.sigma = np.random.uniform(0.5, 1)  # standard deviation

    def sample(self):
        return np.random.normal(self.mu, self.sigma) # Return a random value with the specified mu and sigma
        
        


# The environment has to inherit the interface of gym.Env
class GamblingRoom(gym.Env):
    """
    A k-armed bandit environment: a gambling room with slot machines, allows the agents to interact with it.
        r_machines: A list of slot machines, each gamblingRoom contains k number of slotMachines
    """
    def __init__(self, k):
        # initialize reward distribution for each action/machine
        self.r_machines = []
        for i in range(k):
            # each gamblingRoom contains k number of slotMachines
            self.r_machines.append(slotMachine())

        self.num_arms = k
        self.action_space = spaces.Discrete(self.num_arms)
        self.observation_space = spaces.Discrete(1)
        # for our bandit environment, the state is constant
        self.state = 0
        self.seed()
    
    # step up the environment based on the selected action,
    # return the constant state, reward, done = false, and info
    def step(self, action):
      assert self.action_space.contains(action)
      done = False
      
      #the reward is drawn from one of the machines based on the taken action 
      self.state = 0
      reward = self.r_machines[action].sample()
      return self.state, reward, done, {} 

    def seed(self, seed=None):
      pass
    
    def reset(self):
      pass

    def render(self, mode = 'human', close = False):
      pass 

    def close(self):
      pass
    
import random
class EplisonGreedyAgent:
  def __init__(self, k, e):
    # set up the number of arms/actions
    self.num_arms = k
    # set up the value of epsilon
    self.epsilon = e
    # init the estimated values of all actions
    self.Qvalues = np.zeros(k)
    # init the numbers of time step that every action is selected
    self.stepSize = np.zeros(k)

    ##
    # select the action to take at the current time step
    # (for MDP, choose the action based on state; for k-armed bandit, no state given)
    # return: the action to take
    ##
  def select_action(self):
    ########## TODO: to be filled. ##########  
    r = random.random()
    if r > self.epsilon:
      # in case there are more than one max value the fuction np.argam return only the first on in the vector
      return np.argmax(self.Qvalues)
    else:
      return random.randrange(len(self.Qvalues))

    ##
    # Update the Q-values of the agent based on received rewards
    # input: action_index = the action, reward = the reward from this action
    # return: null
    ##
  def update_parameters(self, action, reward)
     # We implement here the action-value update rule for the environment
    self.stepSize[action] += 1
    self.Qvalues[action] += (reward-self.Qvalues[action])/self.stepSize[action] ####
     
      
num_action = 10
num_seed = 5
num_runs = 100  # number of simulation runs
num_episodes = 500  # number of steps in each run
epsilon = 0.01

def run_env(num_action, num_seed, num_runs, num_episodes, epsilon):
  # set up the random seed
  np.random.seed(num_seed)

  # init the environment
  env = GamblingRoom(num_action)
  mu = []
  sigma = []
  for i in range(num_action):
    mu.append(env.r_machines[i].mu)
    sigma.append(env.r_machines[i].sigma)
    #print(i, ": ", (env.r_machines[i].mu, env.r_machines[i].sigma))

  max_index = mu.index(max(mu))
  # delete the wrap
  env = env.unwrapped

  # show the action space
  #print(env.action_space) 

  # run multiple simulations
  run_rewards = np.zeros(num_episodes)
  run_optimal_action = np.zeros(num_episodes)
  for i_run in range(num_runs):
    # init the epsilon-greedy RL agent 
    agent = EplisonGreedyAgent(num_action, epsilon)
    # in each simulation run, loop the action selection
    episode_rewards = []
    episode_optimal_action = []
    for i_step in range(num_episodes):
      action = agent.select_action()
      state, reward, done, info = env.step(action)
      agent.update_parameters(action, reward)
      episode_rewards.append(reward)
      if action == max_index:
        episode_optimal_action.append(1) 
      else:
        episode_optimal_action.append(0)

    # save the result variables you need
    # these two vectors contain resectuvely the sum of the rewards for each episode and the total number that action ar taken
    run_rewards = [x + y for x, y in zip(run_rewards, episode_rewards)] 
    run_optimal_action = [x + y for x, y in zip(run_optimal_action, episode_optimal_action)] 


    
  
  print('sd mean {}'.np.std(mu))
  env.close()
  # the values in run_rewards and run_optimal_action are averaged per numeber of runs
  return [x / num_runs for x in run_rewards], [x / num_runs for x in run_optimal_action], mu, sigma
        
