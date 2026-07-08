The agents are both uploaded as python notebook files, with their respective plotting functions.
In order to run on a local device, make sure that the notebooks are in the same folder as config.json, and env_raw.py,
a slightly modified version of the original env.py to make sure that the agent observes the raw syndromes 
instead of the change in syndromes.
Naturally, the parameters in config.json can be modified for future studies, and the reward formulation in env_raw.py.
Running the agents requires no special steps, only the requirements to be downloaded. 
Due to long run-times, it is recommended to convert the notebooks to python files:
```jupyter nbconvert --to script PPO.ipynb```
Or alternatively
```jupyter nbconvert --to script Q-learning.ipynb```
And to the run it on the LIACS (or alternative) ssh, using a larger number of CPU workers.