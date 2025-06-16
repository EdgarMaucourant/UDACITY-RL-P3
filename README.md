# Project Details
This repository hosts the submited project of Edgar Maucourant for the grading project of the Multi-agent module of the Deep Reinforcement Learning Nanodegree at Udacity.

It uses Unity ML-Agents and Python scripts to train an agent and environment that is simulating a kind of tennis game where each player should never let the ball going down. This is a collaborative game as players should try to keep the ball going back and forth as long as possible, each time a player send the ball over the net it receives a reward of 0.1, otherwise is receive a reward of -0.01.

The agent and environment are provided as a Unity "Game" called Tennis.

Here are the details of the environment

| Type				| Value		|
|-------------------|-----------|
| Action Space      |  2        |
| Observation Shape |  (8,)     |
| Solving score     |  0.5      | 

Here is an example video of an agent trained for 1000 iterations able to solve the experiment with a score of 2.6 :

<video width="1282" height="756" controls>
  <source src="P3_Tennis.mp4" type="video/mp4">
</video>

Please follow the instructions below to train your agents using this repo. Also please look into the [Report](Report.md) file to get more info about how the code is structured and how the model behave under training.

# Getting Started

Before training your model, you need to download and create some elements.

*Note:*  this repo assume that your are running the code on a Windows machine (the Unity game is only provided for Windows) however adapting it to run on Mac or Linux should only require to update the path the the game executable, this has not been tested though.

## Create a Conda env
1. To be able to run the training on a GPU install Cuda 11.8 from (https://developer.nvidia.com/cuda-11-8-0-download-archive)

2. Create (and activate) a new environment with Python 3.9.

```On a terminal
conda create --name drlnd python=3.9 
conda activate drlnd
```
	
3. Install the dependency (only tested on Windows, but should work on other env as well):
```bash
git clone https://github.com/EdgarMaucourant/UDACITY-rl-p3
pip install .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment. 
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

## Instructions to train the agent

To train the agent, please follow the instruction under the section *4. It's Your Turn!* of the Tennis Jupyter Notebook.

1. From a new terminal, open the notebook

```
jupyter notebook Tennis.ipynb
```

2. Make sure your change the kernel to the drlnd kernel created before, use the Kernel menu and click "Change Kernel" then choose drlnd

3. Scroll to the section *4. It's Your Turn!* and run the cell importing the dependencies then the one defining the function "maddpg". This function is used to train the agent using the hyperparameters provided. Note that in our cases we used the default parameters for Number of episodes (1000). Note also that the max steps is not used in the final code.

4. Run the next cell to import the required dependencies, and create a new environment based on the Tennis game (this is where you want to update the reference to the executable if you don't run on Windows). 

This cell also create the agents to be trained, the agents are based on the DDPG Algorithm and expect the state size and action size as input (plus a seed for randomizing the initialization). For more details about this agent please look at the [Report](Report.md).

5. Run the next cell to start the training. After some time (depending on your machine, mine took about 2 hours), your model will be trained and the scores over iterations will be plotted. While training you should see the game running (on windows at least) and the score increasing. 
If after 500 iterations the score did not increase you might want to review the parameters you provided to the ddpg agent (see the [ddpg_agent.py](ddpg_agent.py)).

*Note:* the code expect an average of "0.5" as a score over the last 100 attempts. It is based on the requirement of the project.

## Instructions to see the agent playing

The last cell in the Jupyter notebook shows how to run one episode with a model trained (the pre-trained weights are provided), if you run the cells (after having imported the dependency and created the env, see step 3 and 4 above) you should be able to see the game played by the agent (if you run this code locally on a Windows machine). See how much you agent can get! The videos at the top of this document shows the agent running with the pre-trained weights provided achieving a score of 2.6.
