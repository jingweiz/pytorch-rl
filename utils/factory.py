from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.envs.gym import GymEnv
from core.envs.atari_ram import AtariRamEnv
from core.envs.atari import AtariEnv
from core.envs.lab import LabEnv
from core.envs.opensim import OpenSim
EnvDict = {"gym":       GymEnv,                 # classic control games from openai w/ low-level   input
           "atari-ram": AtariRamEnv,            # atari integrations from openai, with low-level   input
           "atari":     AtariEnv,               # atari integrations from openai, with pixel-level input
           "lab":       LabEnv,
	   "opensim": OpenSim}

from core.models.empty import EmptyModel
from core.models.dqn_mlp import DQNMlpModel
from core.models.dqn_cnn import DQNCnnModel
from core.models.a3c_mlp_con import A3CMlpConModel
from core.models.a3c_cnn_dis import A3CCnnDisModel
from core.models.acer_mlp_dis import ACERMlpDisModel
from core.models.acer_cnn_dis import ACERCnnDisModel
ModelDict = {"empty":        EmptyModel,        # contains nothing, only should be used w/ EmptyAgent
             "dqn-mlp":      DQNMlpModel,       # for dqn low-level    input
             "dqn-mlp-con":  DQNMlpModel,       # for dqn low-level    input
             "dqn-cnn":      DQNCnnModel,       # for dqn pixel-level  input
             "a3c-mlp-con":  A3CMlpConModel,    # for a3c low-level    input (NOTE: continuous must end in "-con")
             "a3c-cnn-dis":  A3CCnnDisModel,    # for a3c pixel-level  input
             "acer-mlp-dis": ACERMlpDisModel,   # for acer low-level   input
             "acer-cnn-dis": ACERCnnDisModel,   # for acer pixel-level input
             "none":         None}

from core.memories.sequential import SequentialMemory
from core.memories.episode_parameter import EpisodeParameterMemory
from core.memories.episodic import EpisodicMemory
MemoryDict = {"sequential":        SequentialMemory,        # off-policy
              "episode-parameter": EpisodeParameterMemory,  # not in use right now
              "episodic":          EpisodicMemory,          # on/off-policy
              "none":              None}                    # on-policy

from core.agents.empty import EmptyAgent
from core.agents.dqn   import DQNAgent
from core.agents.a3c   import A3CAgent
from core.agents.acer  import ACERAgent
AgentDict = {"empty": EmptyAgent,               # to test integration of new envs, contains only the most basic control loop
             "dqn":   DQNAgent,                 # dqn  (w/ double dqn & dueling as options)
             "a3c":   A3CAgent,                 # a3c  (multi-process, pure cpu version)
             "acer":  ACERAgent}                # acer (multi-process, pure cpu version)
