# **Deep Reinforcement Learning** with
# **pytorch** & **visdom**
*******


* Sample testings of trained agents (DQN on Breakout, A3C on Pong, DoubleDQN on CartPole, continuous A3C on InvertedPendulum(MuJoCo)):
<table>
  <tr>
    <td><img src="/assets/breakout.gif?raw=true" width="200"></td>
    <td><img src="/assets/a3c_pong.gif?raw=true" width="200"></td>
    <td><img src="/assets/cartpole.gif?raw=true" width="200"></td>
    <td><img src="/assets/a3c_con.gif?raw=true" width="200"></td>
  </tr>
</table>

* Sample on-line plotting while training an A3C agent on Pong (with 16 learner processes):
![a3c_pong_plot](/assets/a3c_pong.png)

* Sample loggings while training a DQN agent on CartPole (we use ```WARNING``` as the logging level currently to get rid of the ```INFO``` printouts from visdom):
```bash
[WARNING ] (MainProcess) <===================================>
[WARNING ] (MainProcess) bash$: python -m visdom.server
[WARNING ] (MainProcess) http://localhost:8097/env/daim_17040900
[WARNING ] (MainProcess) <===================================> DQN
[WARNING ] (MainProcess) <-----------------------------------> Env
[WARNING ] (MainProcess) Creating {gym | CartPole-v0} w/ Seed: 123
[INFO    ] (MainProcess) Making new env: CartPole-v0
[WARNING ] (MainProcess) Action Space: [0, 1]
[WARNING ] (MainProcess) State  Space: 4
[WARNING ] (MainProcess) <-----------------------------------> Model
[WARNING ] (MainProcess) MlpModel (
  (fc1): Linear (4 -> 16)
  (rl1): ReLU ()
  (fc2): Linear (16 -> 16)
  (rl2): ReLU ()
  (fc3): Linear (16 -> 16)
  (rl3): ReLU ()
  (fc4): Linear (16 -> 2)
)
[WARNING ] (MainProcess) No Pretrained Model. Will Train From Scratch.
[WARNING ] (MainProcess) <===================================> Training ...
[WARNING ] (MainProcess) Validation Data @ Step: 501
[WARNING ] (MainProcess) Start  Training @ Step: 501
[WARNING ] (MainProcess) Reporting       @ Step: 2500 | Elapsed Time: 5.32397913933
[WARNING ] (MainProcess) Training Stats:   epsilon:          0.972
[WARNING ] (MainProcess) Training Stats:   total_reward:     2500.0
[WARNING ] (MainProcess) Training Stats:   avg_reward:       21.7391304348
[WARNING ] (MainProcess) Training Stats:   nepisodes:        115
[WARNING ] (MainProcess) Training Stats:   nepisodes_solved: 114
[WARNING ] (MainProcess) Training Stats:   repisodes_solved: 0.991304347826
[WARNING ] (MainProcess) Evaluating      @ Step: 2500
[WARNING ] (MainProcess) Iteration: 2500; v_avg: 1.73136949539
[WARNING ] (MainProcess) Iteration: 2500; tderr_avg: 0.0964358523488
[WARNING ] (MainProcess) Iteration: 2500; steps_avg: 9.34579439252
[WARNING ] (MainProcess) Iteration: 2500; steps_std: 0.798395631184
[WARNING ] (MainProcess) Iteration: 2500; reward_avg: 9.34579439252
[WARNING ] (MainProcess) Iteration: 2500; reward_std: 0.798395631184
[WARNING ] (MainProcess) Iteration: 2500; nepisodes: 107
[WARNING ] (MainProcess) Iteration: 2500; nepisodes_solved: 106
[WARNING ] (MainProcess) Iteration: 2500; repisodes_solved: 0.990654205607
[WARNING ] (MainProcess) Saving Model    @ Step: 2500: /home/zhang/ws/17_ws/pytorch-rl/models/daim_17040900.pth ...
[WARNING ] (MainProcess) Saved  Model    @ Step: 2500: /home/zhang/ws/17_ws/pytorch-rl/models/daim_17040900.pth.
[WARNING ] (MainProcess) Resume Training @ Step: 2500
...
```
*******


## What is included?
This repo currently contains the following agents:

- Deep Q Learning (DQN) [[1]](http://arxiv.org/abs/1312.5602), [[2]](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)
- Double DQN [[3]](http://arxiv.org/abs/1509.06461)
- Dueling network DQN (Dueling DQN) [[4]](https://arxiv.org/abs/1511.06581)
- Asynchronous Advantage Actor-Critic (A3C) (w/ both discrete/continuous action space support) [[5]](https://arxiv.org/abs/1602.01783), [[6]](https://arxiv.org/abs/1506.02438)

Work in progress:
- Deep Deterministic Policy Gradient (DDPG) [[7]](http://arxiv.org/abs/1509.02971)
- Continuous DQN (CDQN or NAF) [[8]](http://arxiv.org/abs/1603.00748)


## Code structure & Naming conventions:
NOTE: we follow the exact code structure as [pytorch-dnc](https://github.com/jingweiz/pytorch-dnc) so as to make the code easily transplantable.
* ```./utils/factory.py```
> We suggest the users refer to ```./utils/factory.py```,
 where we list all the integrated ```Env```, ```Model```,
 ```Memory```, ```Agent``` into ```Dict```'s.
 All of those four core classes are implemented in ```./core/```.
 The factory pattern in ```./utils/factory.py``` makes the code super clean,
 as no matter what type of ```Agent``` you want to train,
 or which type of ```Env``` you want to train on,
 all you need to do is to simply modify some parameters in ```./utils/options.py```,
 then the ```./main.py``` will do it all (NOTE: this ```./main.py``` file never needs to be modified).
* namings
> To make the code more clean and readable, we name the variables using the following pattern (mainly in inherited ```Agent```'s):
> * ```*_vb```: ```torch.autograd.Variable```'s or a list of such objects
> * ```*_ts```: ```torch.Tensor```'s or a list of such objects
> * otherwise: normal python datatypes


## Dependencies
- Python 2.7
- [PyTorch](http://pytorch.org/)
- [Visdom](https://github.com/facebookresearch/visdom)
- [OpenAI Gym](https://github.com/openai/gym)
- [mujoco-py (Optional: for training continuous version of a3c)](https://github.com/openai/mujoco-py)
*******


## How to run:
You only need to modify some parameters in ```./utils/options.py``` to train a new configuration.

* Configure your training in ```./utils/options.py```:
> * ```line 14```: add an entry into ```CONFIGS``` to define your training (```agent_type```, ```env_type```, ```game```, ```model_type```, ```memory_type```)
> * ```line 33```: choose the entry you just added
> * ```line 29-30```: fill in your machine/cluster ID (```MACHINE```) and timestamp (```TIMESTAMP```) to define your training signature (```MACHINE_TIMESTAMP```),
 the corresponding model file and the log file of this training will be saved under this signature (```./models/MACHINE_TIMESTAMP.pth``` & ```./logs/MACHINE_TIMESTAMP.log``` respectively).
 Also the visdom visualization will be displayed under this signature (first activate the visdom server by type in bash: ```python -m visdom.server &```, then open this address in your browser: ```http://localhost:8097/env/MACHINE_TIMESTAMP```)
> * ```line 32```: to train a model, set ```mode=1``` (training visualization will be under ```http://localhost:8097/env/MACHINE_TIMESTAMP```); to test the model of this current training, all you need to do is to set ```mode=2``` (testing visualization will be under ```http://localhost:8097/env/MACHINE_TIMESTAMP_test```).

* Run:
> ```python main.py```
*******


## Repos we referred to during the development of this repo:
* [matthiasplappert/keras-rl](https://github.com/matthiasplappert/keras-rl)
* [transedward/pytorch-dqn](https://github.com/transedward/pytorch-dqn)
* [ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
* [onlytailei/A3C-PyTorch](https://github.com/onlytailei/A3C-PyTorch)
* And a private implementation of A3C from [@stokasto](https://github.com/stokasto)
