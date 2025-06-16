# Project Details
This repository hosts the submited project of Edgar Maucourant for the grading project of the Policy-Based Methods module of the Deep Reinforcement Learning Nanodegree at Udacity.

It uses Unity ML-Agents and Python scripts to train an agent and environment that is simulating a two parts robotic arm to reach a target. Each time the arm is able to aim to the moving target it receives a reward of 1, othwerwise it receives 0.

The agent and environment are provided as a Unity "Game" called Reacher.

Note that this environment uses a single arm, another environment includes 20 arms to train faster, however this has not been used in this experiment (see the [Report](Report.md) for future improvements).

Here are the details of the environment

| Type				| Value		|
|-------------------|-----------|
| Action Space      |  4        |
| Observation Shape |  (33,)    |
| Solving score     |  30       | 

Here is an example video of an agent trained for 1000 iterations able to solve the experiment with a score of 39 :

<video width="1292" height="784" controls>
  <source src="P2_Reacher.mp4" type="video/mp4">
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
git clone https://github.com/EdgarMaucourant/udacity-rl-p2
pip install .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment. 
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

## Instructions to train the agent

To train the agent, please follow the instruction under the section *4. It's Your Turn!* of the Continuous_Control Jupyter Notebook.

1. From a new terminal, open the notebook

```
jupyter notebook Continuous_Control.ipynb
```

2. Make sure your change the kernel to the drlnd kernel created before, use the Kernel menu and click "Change Kernel" then choose drlnd

3. Scroll to the section *4. It's Your Turn!* and run the cell importing the dependencies then the one defining the function "ddpg". This function is used to train the agent using the hyperparameters provided. Note that in our cases we used the default parameters for Number of episodes (1000) and max steps (1000). Note also that the max steps is not used in the final code.

4. Run the next cell to import the required dependencies, and create a new environment based on the Reacher game (this is where you want to update the reference to the executable if you don't run on Windows). 

This cell also create the Agent to be trained, that agent is based on the DDPG Algorithm and expect the state size and action size as input (plus a seed for randomizing the initialization). For more details about this agent please look at the [Report](Report.md).

5. Run the next cell to start the training. After some time (depending on your machine, mine took about 2 hours), your model will be trained and the scores over iterations will be plotted. While training you should see the game running (on windows at least) and the score increasing. 
If after 500 iterations the score did not increase you might want to review the parameters you provided to the ddpg agent (see the [ddpg_agent.py](ddpg_agent.py)).

*Note:* the code expect an average of "30" as a score over the last 100 attempts. It is based on the requirement of the project.

## Instructions to see the agent playing

The last cell in the Jupyter notebook shows how to run one episode with a model trained (the pre-trained weights are provided), if you run the cells (after having imported the dependency and created the env, see step 3 and 4 above) you should be able to see the game played by the agent (if you run this code locally on a Windows machine). See how much you agent can get! The videos at the top of this document shows the agent running with the pre-trained weights provided achieving a score of XX.
