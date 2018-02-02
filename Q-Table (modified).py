'''
Solving OpenAI gym's FrozenLake environment with a Q-table.
A description of the FrozenLake environment can be found here: https://gym.openai.com/envs/FrozenLake-v0/
The goal is to obtain a frisbee located on a frozen lake, which is pocketed with holes.
The environment consists of a 4x4 grid:
    S  F  F  F
    F  H  F  H
    F  F  F  H
    H  F  F  G
Where   S: starting point, safe
        F: frozen surface, safe
        H: hole, fall to your doom
        G: goal, where the frisbee is located
'''

import gym
import numpy as np

'''Load the environment'''
env = gym.make('FrozenLake-v0')


'''Implement Q-Table learning algorithm'''
# Initialize a table with all zeros
'''
The table size is 16x4: 16 observable states (as the environment is a 4x4 grid), and
                        4 possible actions (move: up, down, left, or right).
The states are linearised into the rows of the Q-table, while the columns define the actions.
Each entry defines the reward for perfoming the action in the column, at the location in the row.
So the Q-table looks like this:
             L   D   R   U
1  (1,1) S:  x   x   x   x
2  (1,2) F:  x   x   x   x
3  (1,3) F:  x   x   x   x
4  (1,4) F:  x   x   x   x
5  (2,1) F:  x   x   x   x
6  (2,2) H:  x   x   x   x
7  (2,3) F:  x   x   x   x
8  (2,4) H:  x   x   x   x
9  (3,1) F:  x   x   x   x
10 (3,2) F:  x   x   x   x
11 (3,3) F:  x   x   x   x
12 (3,4) H:  x   x   x   x
13 (4,1) H:  x   x   x   x
14 (4,2) F:  x   x   x   x
15 (4,3) F:  x   x   x   x
16 (4,4) G:  x   x   x   x
See: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
'''
Q = np.zeros([env.observation_space.n,env.action_space.n]) # Q-table matrix

# Set learning hyperparameters
lr = .8 # learning rate coefficient: this defines how quickly/slowly the agent learns. (We don't have to update the Q-table with the full value of the Bellman equation, it's better to learn slower.)
gamma = .95 # discount factor: this defines how short/long-term oriented the agent is
num_episodes = 2000 # number of times to run the simulation and update the Q-table

# create list to contain total reward per episode
rList = [] # list of rewards

for i in range(num_episodes): # run the simulation for num_episodes
    # Initialise; reset environment and get first new observation
    s = env.reset() # reset the gym environment and obtain the initial state
    rTotal = 0 # initialise a variable equal to the total sum of the rewards
    done = False # initialise boolean instructing whether the episode has terminated or not
    j = 0 # initialise number of steps taken before termination of the episode

    # The Q-Table learning algorithm
    while j < 99: # 99 is an arbitrarily large number for the maximum number steps to take
        j+=1 # increase the number of steps by one

        # Choose an action by greedily (with noise) picking from Q table
        '''
        np.argmax(Q[s,:]) chooses the action which corresponds to the maximum Q-value in the Q-table

        np.random.randn(1,env.action_space.n) generates the noise: an array of shape (1, action_space) – (1, 4), filled with
        random floats sampled from a univariate “normal” (Gaussian) distribution, of mean 0 and variance 1.

        Multiplying by (1./(i+1)) ensures that the noise decreases as the number of episodes increases – thus over time the
        agent tends to be more consistent and less random in its choice of actions.
        '''
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1))) # a* = argmax_a(Q(s,a) + ε)

        # Get new state and reward from environment
        s1,r,done,_ = env.step(a) # env.step(a) function returns the result of the action in 4 values: observation (object), reward (float), done (boolean), info (dict). See https://gym.openai.com/docs/ Observations section. We will ignore the info value.

        # Update Q-Table with new knowledge
        '''
        Q(s,a) = r + γ*max(Q(s',a')) <=  This is the Bellman equation:
                                         The immediate reward plus the discounted present value of future rewards.

        To update the Q-values:
            Find the difference between this new Q-value and the previous Q-value (= 0 if there's no change)
                Δ = (r + γ*max(Q(s',a'))) - Q(s,a)
            Modify this difference with a learning rate (α) to ensure the agent doesn't learn "too fast"
                αΔ = α[ r + γ*max(Q(s',a')) - Q(s,a) ]
            Update our Q-table values with the value we had before, plus the difference
                Q'(s,a) = Q(s,a) + αΔ = Q(s,a) + α[ r + *max(Q(s',a')) - Q(s,a) ]
        '''
        Q[s,a] = Q[s,a] + lr*(r + gamma*np.max(Q[s1,:]) - Q[s,a]) # Recompute Q-values using the Bellman equation
        rTotal += r # add reward obtained in this state to the total reward
        s = s1 # set the state to the next state
        if done == True:
            break # exit the while loop and begin a new episode if done

    rList.append(rTotal) # list of the total reward receieved per episode

'''Display the results of the episodes'''
# Print score over time
print("Score over time: " +  str(sum(rList)/num_episodes)) # prints the average reward receieved for all episodes (range = [0:1])

# Print the final Q-Table values
print("Final Q-Table Values")
print(Q) # Display the final Q-table


''''Print the final greedy path'''
s = env.reset() # reset the gym environment and obtain the initial state
done = False # initialise boolean instructing whether the episode has terminated or not
j = 0 # initialise number of steps taken before termination of the episode
path = [0]

# The Q-Table learning algorithm
while j < 99: # 99 is an arbitrarily large number of maximum steps to take
    j+=1 # increase the number of steps by one

    # Choose an action by greedily picking from Q table
    a = np.argmax(Q[s,:]) # a* = argmax_a(Q(s,a))

    # Get new state and reward from environment
    s1,r,done,_ = env.step(a) # env.step(a) function returns the result of the action in 4 values: observation (object), reward (float), done (boolean), info (dict). See https://gym.openai.com/docs/ Observations section. We will ignore the info value.
    s = s1 # set the state to the next state
    path.append(s)
    if done == True:
        break # exit the while loop and begin a new episode if done
print(path)
