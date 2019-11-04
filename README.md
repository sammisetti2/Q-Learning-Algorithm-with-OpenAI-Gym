# Q-Learning-Algorithm-with-OpenAI-Gym
A college project to understand the working of a Q-learning algorithm on the OpenAI Gym environments: FrozenLake-V0 and Taxi-v3. It also outputs the environments to observe how the trained agent plays through the environments.

# Usage
OpenAI Gym doesn't render the environments properly, specifically the colors and blocks, in Windows OS. It's recommended to run this on a Mac OS or Linux OS, like Ubuntu, to render it properly. This program is run in the terminal. Make sure you have Python 3.5 or higher installed on your computer.

1. Download/clone the source code onto your computer.

2. In the terminal, install gym using "pip install gym".

3. Next, install IPython using "pip install ipython".

4. Finally, run each file individually in the terminal.

Each file is the implementation of the same Q-learning algorithm for the respective environments, FrozenLake-v0 and Taxi-v3. When running each file, the average reward for every thousand episodes, till 10,000, is outputted. Following that, the Q-table tabulated, after the agent is trained for 10,000 episodes, is printed. Finally, it will render the trained agent playing through the environment for a certain number of episodes (which you can change in the code as per your interest). 

Side Note: The rewards in FrozenLake-v0 are +1 for reaching the goal and 0 for falling in a hole. The rewards in Taxi-v3 are +20 points for a successful dropoff, -1 for each timestep, and -10 for illegal dropoff/pickup.

The links for the source code of the environments are:
https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
