from __future__ import absolute_import
from __future__ import division

from rl.env import GymEnv, AtariRamEnv, AtariEnv, LabEnv, GazeboEnv
EnvDict = {"gym":       GymEnv,                 # classic control games from openai w/ full-state-vector as input
           "atari-ram": AtariRamEnv,            # atari integrations from openai, with full-state-vector as input
           "atari":     AtariEnv,               # atari integrations from openai, with pixel-level input
           "lab":       LabEnv,
           "gazebo":    GazeboEnv}              # GazeboEnv

from rl.model import MlpModel, CnnModel, A3CMlpModel, A3CCnnModel, A3CGazModel, A3CMjcModel
ModelDict = {"mlp":     MlpModel,               # for full-state  input envs
             "cnn":     CnnModel,               # for pixel-level input envs
             "a3c-mlp": A3CMlpModel,            # for full-state  input envs
             "a3c-cnn": A3CCnnModel,            # for pixel-level input envs
             "a3c-gaz": A3CGazModel,            # for gazebo continuous input envs
             "a3c-mjc": A3CMjcModel}            # for mujoco continuous

from rl.memory import SequentialMemory
MemoryDict = {"sequential": SequentialMemory,   # off-policy
              "none":       None}               #  on-policy

from rl.agents.empty import EmptyAgent
from rl.agents.dqn   import DQNAgent
from rl.agents.a3c   import A3CAgent
AgentDict = {"empty": EmptyAgent,               # to test integration of new envs, contains only the most basic control loop
             "dqn":   DQNAgent,                 # dqn (w/ double dqn & dueling as options)
             "a3c":   A3CAgent}                 # a3c (multi-process, pure cpu version)
