import gym
import numpy
import random
from time import sleep
from os import system, name

env = gym.make("Taxi-v3").env

def clear(): 
  
    # Clear on Windows.
    if name == 'nt': 
        _ = system('cls')
  
    # Clear on Mac and Linux. (os.name is 'posix') 
    else: 
        _ = system('clear')

clear()
# states = 500 (5 * 5 environment * 4 destination * 5 passenger locations)
# actions = 6 (south , north , east , west , pickup , dropoff)

q_table = numpy.zeros([env.observation_space.n, env.action_space.n]) # Creating our Q-table

training_times = 30000
display_times = 5

alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []

for train in range(training_times):
        state = env.reset()
        epochs, penalties, reward, = 0, 0, 0
        done  = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = numpy.argmax(q_table[state]) # Pick the action which has previously given the highest reward.

            next_state, reward, done, info = env.step(action) 
            
            old_value = q_table[state, action] # Retrieve old value from the q-table.
            next_max = numpy.max(q_table[next_state])

            # Update q-value 
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10: # Checks illegal action.
                penalties += 1

            state = next_state
        if train % 1000 == 0: # Output number of completed episodes every 1000 episodes.
            print(f"Episode: {train}")

print("Training finished.\n")
sleep(1)



total_epochs, total_penalties = 0, 0


for display in range(display_times):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = numpy.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        clear()
        env.render()
        print("Timestep: " + str(epochs))
        print("State: " + str(state))
        print("Action: "  + str(action))
        print("Reward: " + str(reward))
        sleep(0.2) 

    total_penalties += penalties
    total_epochs += epochs

print("Results after " + str(display_times) +  " episodes:")
print("Average timesteps per episode: " + str(total_epochs / display_times))
print("Average penalties per episode: " + str(total_penalties / display_times))
