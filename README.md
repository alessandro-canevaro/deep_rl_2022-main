# Deep learning project: Reinforcement learning

Value iteration networks represent a way to combine reinforcement learning with planning. The idea is fairly simple: If we know a model of the environment, it is well known that the so-called Value-iteration algorithm (See Suttons book linked above) is the optimal way to plan ahead. A ‘model’ in this context is essentially just a mapping from inputs to the reinforcement learning agent to a probability distribution, i.e. you can think about it as a softmax multi-class classifier. 
In reinforcement learning, we don’t know a model, but we know how to build a softmax classifier. So what that gives us is: 
 - We build a softmax classifier, which output a possible model of the environment
 - Given this model, we can compute the optimal action by running the value-iteration algorithm
 - The reinforcement learning agent can then output this action
 
The idea is pretty simple: When we interact with the environment, we get information about how good or bad an action is. We can use this information to train the neural network using gradient descent. The neat thing is that the combined algorithm ends up resembling a standard convolution neural network. Since we are integrating planning with the reinforcement learning algorithm, this makes the combined model able to generalize to new situations. You can read more in the original paper: 
https://arxiv.org/abs/1602.02867

# Background and support
 - Project description here: https://docs.google.com/document/d/1ChgaSyp1QnZfhBaXc6Bna248UmY43o7wVpFI9FL6U5M/edit
 - Discord here: https://discord.gg/dCYTJsyUCf
 - Videos can be found here: https://video.dtu.dk/channel/Deep%2BLearning%2B2022%253A%2BThe%2BRL%2Bproject/555910

# Useful commands for set up
 - Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
 - .\\[your env name]\Scripts\activate
 - python -m pip install -r requirements.txt
 - pip install -e .
 
# Important files
 - the entry points of the project are located in the raya3c folder
 - my_vpn.py contains the VPNNetwork custom model with the value propagation implementation
 - my_vpn_mvprop.py contains the VPNNetwork custom model with the maximum value propagation implementation
 - train_my_vpn.py is used to train the network
 - test_my_vpn.py is used to test the network
 - maze_register.py is used for creating custom environments

# How to run:
 - for training, set up the paramenters in the experiments_config.py file, then run the train_my_vpn.py
 - for testing, adapt the paramenters in the experiments_config.py to match the saved checkpoint, then run test_my_vpn.py
 - if you want to create a new environment modify the maze_register.py file

# P.S.:
 - Currently the experiments_config.py is set up to run the model with the TESTDEMO checkpoint (trained on a 5x5 maze with walls)
