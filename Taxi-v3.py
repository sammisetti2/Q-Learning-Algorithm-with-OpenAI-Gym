import numpy as np
import gym
import random
import time
from IPython.display import clear_output

#environment is being taken from gym
env = gym.make("Taxi-v3")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

#initializing required variables
q_table = np.zeros((state_space_size, action_space_size))

#Can be varied to see how the agent tries to play the game later on with lesser experience
num_episodes = 10000
max_steps_per_episode = 50

#Using this, changes the rate at which the agent learns and how much it should look into the future for rewards
learning_rate = 0.1
discount_rate = 0.80

#Determines the rate at which the agent explore's the environment
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

#Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        #Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)

        #Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if done == True:
            break

    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)

#Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
print("Average reward per thousand episodes\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

#Print updated Q-table
print("\n\n Q-table \n")
print(q_table)


#Watch Agent play Taxi with best action
#outer for loop is for each episode
for episode in range(20):
    state =  env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)
    
    #Inner for loop is for each step/action in the episode
    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        
        #choosing the best action from Q-table
        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)
        
        #Checking if chosen action is pickup or dropoff
        if ((action == 5) or (action == 4)):
            clear_output(wait=True)
            env.render()

            #If pickup, then check if it's illegal pickup or not
            if(action == 4):
                #Each state is the possible combination of pickup and dropoff locations
                if ((new_state == 19) or (new_state == 99) or (new_state == 417) or (new_state == 477) or (new_state == 419) or (new_state == 416) or (new_state == 17) or (new_state == 478) or (new_state == 18) or (new_state == 96) or (new_state == 476) or (new_state == 98)):
                    state = new_state
                    continue
                else:
                    print("****Illegal Action****")
                    time.sleep(3)
                    clear_output(wait=True)
                    break
            #If dropoff, then check if it's illegal or not
            if(action == 5):
                #Each state is according to the four possible destinations
                if ((new_state == 85) or (new_state == 0) or (new_state == 410) or (new_state == 475)):
                    print("****Successful Dropoff!****")
                    time.sleep(3)
                else:
                    print("****Illegal Action****")
                    time.sleep(3)
                    clear_output(wait=True)
                break

        state = new_state

env.close()
