import numpy as np

# custom modules
from utils.options import Options
from utils.factory import EnvDict, ModelDict, MemoryDict, AgentDict

# 0. setting up
opt = Options()
np.random.seed(opt.seed)

# 1. env    (prototype)
env_prototype    = EnvDict[opt.env_type]
# 2. model  (prototype)
model_prototype  = ModelDict[opt.model_type]
# 3. memory (prototype)
memory_prototype = MemoryDict[opt.memory_type]
# 4. agent
agent = AgentDict[opt.agent_type](opt.agent_params,
                                  env_prototype    = env_prototype,
                                  model_prototype  = model_prototype,
                                  memory_prototype = memory_prototype)
# 5. fit model
if opt.mode == 1:   # train
    agent.fit_model()
elif opt.mode == 2: # test opt.model_file
    agent.test_model()
