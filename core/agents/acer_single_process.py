from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import time
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.helpers import ACER_Experience
from core.agent_single_process import AgentSingleProcess

class ACERSingleProcess(AgentSingleProcess):
    def __init__(self, master, process_id=0):
        super(ACERSingleProcess, self).__init__(master, process_id)

        # NOTE: diff from pure on-policy methods like a3c, acer is capable of
        # NOTE: off-policy learning and can make use of replay buffer
        self.memory = self.master.memory_prototype(capacity = self.master.memory_params.memory_size // self.master.num_processes,
                                                   max_episode_length = self.master.early_stop)

        # lstm hidden states
        if self.master.enable_lstm:
            self._reset_lstm_hidden_vb_episode() # clear up hidden state
            self._reset_lstm_hidden_vb_rollout() # detach the previous variable from the computation graph

        # # NOTE global variable pi
        # if self.master.enable_continuous:
        #     self.pi_vb = Variable(torch.Tensor([math.pi]).type(self.master.dtype))

        self.master.logger.warning("Registered ACER-SingleProcess-Agent #" + str(self.process_id) + " w/ Env (seed:" + str(self.env.seed) + ").")

    # NOTE: to be called at the beginning of each new episode, clear up the hidden state
    def _reset_lstm_hidden_vb_episode(self, training=True): # seq_len, batch_size, hidden_dim
        not_training = not training
        if self.master.enable_continuous:
            # self.lstm_hidden_vb = (Variable(torch.zeros(2, self.master.hidden_dim).type(self.master.dtype), volatile=not_training),
            #                        Variable(torch.zeros(2, self.master.hidden_dim).type(self.master.dtype), volatile=not_training))
            pass
        else:
            self.lstm_hidden_vb = (Variable(torch.zeros(1, self.master.hidden_dim).type(self.master.dtype), volatile=not_training),
                                   Variable(torch.zeros(1, self.master.hidden_dim).type(self.master.dtype), volatile=not_training))

    # NOTE: to be called at the beginning of each rollout, detach the previous variable from the graph
    def _reset_lstm_hidden_vb_rollout(self):
        self.lstm_hidden_vb = (Variable(self.lstm_hidden_vb[0].data),
                               Variable(self.lstm_hidden_vb[1].data))

    def _preprocessState(self, state, is_valotile=False):
        if isinstance(state, list):
            state_vb = []
            for i in range(len(state)):
                state_vb.append(Variable(torch.from_numpy(state[i]).unsqueeze(0).type(self.master.dtype), volatile=is_valotile))
        else:
            state_vb = Variable(torch.from_numpy(state).unsqueeze(0).type(self.master.dtype), volatile=is_valotile)
        return state_vb

    def _forward(self, state_vb):
        if not self.master.enable_continuous:
            if self.master.enable_lstm:
                # p_vb, v_vb, self.lstm_hidden_vb = self.model(state_vb, self.lstm_hidden_vb)
                pass
            else:
                # p_vb, v_vb = self.model(state_vb)
                pass
            if self.training:
                # action = p_vb.multinomial().data[0][0]
                pass
            else:
                # action = p_vb.max(1)[1].data.squeeze().numpy()[0]
                pass
            # return action, p_vb, v_vb
        else:   # NOTE continous control p_vb here is the mu_vb of continous action dist
            if self.master.enable_lstm:
                # p_vb, sig_vb, v_vb, self.lstm_hidden_vb = self.model(state_vb, self.lstm_hidden_vb)
                pass
            else:
                # p_vb, sig_vb, v_vb = self.model(state_vb)
                pass
            if self.training:
                # _eps = torch.randn(p_vb.size())
                # action = (p_vb + sig_vb.sqrt()*Variable(_eps)).data.numpy()
                pass
            else:
                # action = p_vb.data.numpy()
                pass
            # return action, p_vb, sig_vb, v_vb

class ACERLearner(ACERSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> ACER-Learner #" + str(process_id) + " {Env & Model}")
        super(ACERLearner, self).__init__(master, process_id)

        # learning algorithm    # TODO: adjust learning to each process maybe ???
        self.optimizer = self.master.optim(self.model.parameters(), lr = self.master.lr)

        self._reset_rollout()

        self.training = True    # choose actions by polinomial
        self.model.train(self.training)
        # local counters
        self.frame_step   = 0   # local frame step counter
        self.train_step   = 0   # local train step counter
        # local training stats
        self.p_loss_avg   = 0.  # global policy loss
        self.v_loss_avg   = 0.  # global value loss
        self.loss_avg     = 0.  # global value loss
        self.loss_counter = 0   # storing this many losses
        self._reset_training_loggings()

        # copy local training stats to global every prog_freq
        self.last_prog = time.time()

    def _reset_training_loggings(self):
        self.p_loss_avg   = 0.
        self.v_loss_avg   = 0.
        self.loss_avg     = 0.
        self.loss_counter = 0

    def _reset_rollout(self):       # for storing the experiences collected through one rollout
        self.rollout = ACER_Experience(state0 = [],
                                       action = [],
                                       reward = [],
                                       state1 = [],
                                       terminal1 = [],
                                       policy_vb = [])

    def _backward(self):
        pass

    def _rollout(self, episode_steps, episode_reward):
        return episode_steps, episode_reward

    def run(self):
        # make sure processes are not completely synced by sleeping a bit
        time.sleep(int(np.random.rand() * (self.process_id + 5)))

        nepisodes = 0
        nepisodes_solved = 0
        episode_steps = None
        episode_reward = None
        should_start_new = True
        while self.master.train_step.value < self.master.steps:
            # print(self.master.train_step.value)
            # sync in every step
            self._sync_local_with_global()
            self.optimizer.zero_grad()
            # reset rollout experiences
            self._reset_rollout()

            # start of a new episode
            if should_start_new:
                episode_steps = 0
                episode_reward = 0.
                # reset lstm_hidden_vb for new episode
                if self.master.enable_lstm:
                    # NOTE: clear hidden state at the beginning of each episode
                    self._reset_lstm_hidden_vb_episode()
                # Obtain the initial observation by resetting the environment
                self._reset_experience()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                # reset flag
                should_start_new = False
            if self.master.enable_lstm:
                # NOTE: detach the previous hidden variable from the graph at the beginning of each rollout
                self._reset_lstm_hidden_vb_rollout()
            # Run a rollout for rollout_steps or until terminal
            episode_steps, episode_reward = self._rollout(episode_steps, episode_reward)

            if self.experience.terminal1 or \
               self.master.early_stop and episode_steps >= self.master.early_stop:
                nepisodes += 1
                should_start_new = True
                if self.experience.terminal1:
                    nepisodes_solved += 1

            # calculate loss
            self._backward()

            # copy local training stats to global at prog_freq, and clear up local stats
            if time.time() - self.last_prog >= self.master.prog_freq:
                self.master.p_loss_avg.value   += self.p_loss_avg
                self.master.v_loss_avg.value   += self.v_loss_avg
                self.master.loss_avg.value     += self.loss_avg
                self.master.loss_counter.value += self.loss_counter
                self._reset_training_loggings()
                self.last_prog = time.time()

class ACEREvaluator(ACERSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> ACER-Evaluator {Env & Model}")
        super(ACEREvaluator, self).__init__(master, process_id)

        self.training = False   # choose actions w/ max probability
        self.model.train(self.training)
        self._reset_loggings()

        self.start_time = time.time()
        self.last_eval = time.time()

    def _reset_loggings(self):
        # training stats across all processes
        self.p_loss_avg_log = []
        self.v_loss_avg_log = []
        self.loss_avg_log = []
        # evaluation stats
        self.entropy_avg_log = []
        self.v_avg_log = []
        self.steps_avg_log = []
        self.steps_std_log = []
        self.reward_avg_log = []
        self.reward_std_log = []
        self.nepisodes_log = []
        self.nepisodes_solved_log = []
        self.repisodes_solved_log = []
        # placeholders for windows for online curve plotting
        if self.master.visualize:
            # training stats across all processes
            self.win_p_loss_avg = "win_p_loss_avg"
            self.win_v_loss_avg = "win_v_loss_avg"
            self.win_loss_avg = "win_loss_avg"
            # evaluation stats
            self.win_entropy_avg = "win_entropy_avg"
            self.win_v_avg = "win_v_avg"
            self.win_steps_avg = "win_steps_avg"
            self.win_steps_std = "win_steps_std"
            self.win_reward_avg = "win_reward_avg"
            self.win_reward_std = "win_reward_std"
            self.win_nepisodes = "win_nepisodes"
            self.win_nepisodes_solved = "win_nepisodes_solved"
            self.win_repisodes_solved = "win_repisodes_solved"

    def _eval_model(self):
        print("===================================== eval")
        self.last_eval = time.time()
        eval_at_train_step = self.master.train_step.value
        eval_at_frame_step = self.master.frame_step.value
        # first grab the latest global model to do the evaluation
        self._sync_local_with_global()

        # evaluate
        eval_step = 0

        eval_entropy_log = []
        eval_v_log = []
        eval_nepisodes = 0
        eval_nepisodes_solved = 0
        eval_episode_steps = None
        eval_episode_steps_log = []
        eval_episode_reward = None
        eval_episode_reward_log = []
        eval_should_start_new = True
        while eval_step < self.master.eval_steps:
            # TODO:
            eval_step += 1

        # Logging for this evaluation phase
        loss_counter = self.master.loss_counter.value
        p_loss_avg = self.master.p_loss_avg.value / loss_counter if loss_counter > 0 else 0.
        v_loss_avg = self.master.v_loss_avg.value / loss_counter if loss_counter > 0 else 0.
        loss_avg = self.master.loss_avg.value / loss_counter if loss_counter > 0 else 0.
        self.master._reset_training_loggings()
        self.p_loss_avg_log.append([eval_at_train_step, p_loss_avg])
        self.v_loss_avg_log.append([eval_at_train_step, v_loss_avg])
        self.loss_avg_log.append([eval_at_train_step, loss_avg])
        self.entropy_avg_log.append([eval_at_train_step, np.mean(np.asarray(eval_entropy_log))])
        self.v_avg_log.append([eval_at_train_step, np.mean(np.asarray(eval_v_log))])
        self.steps_avg_log.append([eval_at_train_step, np.mean(np.asarray(eval_episode_steps_log))])
        self.steps_std_log.append([eval_at_train_step, np.std(np.asarray(eval_episode_steps_log))]); del eval_episode_steps_log
        self.reward_avg_log.append([eval_at_train_step, np.mean(np.asarray(eval_episode_reward_log))])
        self.reward_std_log.append([eval_at_train_step, np.std(np.asarray(eval_episode_reward_log))]); del eval_episode_reward_log
        self.nepisodes_log.append([eval_at_train_step, eval_nepisodes])
        self.nepisodes_solved_log.append([eval_at_train_step, eval_nepisodes_solved])
        self.repisodes_solved_log.append([eval_at_train_step, (eval_nepisodes_solved/eval_nepisodes) if eval_nepisodes > 0 else 0.])
        # plotting
        if self.master.visualize:
            self.win_p_loss_avg = self.master.vis.scatter(X=np.array(self.p_loss_avg_log), env=self.master.refs, win=self.win_p_loss_avg, opts=dict(title="p_loss_avg"))
            self.win_v_loss_avg = self.master.vis.scatter(X=np.array(self.v_loss_avg_log), env=self.master.refs, win=self.win_v_loss_avg, opts=dict(title="v_loss_avg"))
            self.win_loss_avg = self.master.vis.scatter(X=np.array(self.loss_avg_log), env=self.master.refs, win=self.win_loss_avg, opts=dict(title="loss_avg"))
            self.win_entropy_avg = self.master.vis.scatter(X=np.array(self.entropy_avg_log), env=self.master.refs, win=self.win_entropy_avg, opts=dict(title="entropy_avg"))
            self.win_v_avg = self.master.vis.scatter(X=np.array(self.v_avg_log), env=self.master.refs, win=self.win_v_avg, opts=dict(title="v_avg"))
            self.win_steps_avg = self.master.vis.scatter(X=np.array(self.steps_avg_log), env=self.master.refs, win=self.win_steps_avg, opts=dict(title="steps_avg"))
            # self.win_steps_std = self.master.vis.scatter(X=np.array(self.steps_std_log), env=self.master.refs, win=self.win_steps_std, opts=dict(title="steps_std"))
            self.win_reward_avg = self.master.vis.scatter(X=np.array(self.reward_avg_log), env=self.master.refs, win=self.win_reward_avg, opts=dict(title="reward_avg"))
            # self.win_reward_std = self.master.vis.scatter(X=np.array(self.reward_std_log), env=self.master.refs, win=self.win_reward_std, opts=dict(title="reward_std"))
            self.win_nepisodes = self.master.vis.scatter(X=np.array(self.nepisodes_log), env=self.master.refs, win=self.win_nepisodes, opts=dict(title="nepisodes"))
            self.win_nepisodes_solved = self.master.vis.scatter(X=np.array(self.nepisodes_solved_log), env=self.master.refs, win=self.win_nepisodes_solved, opts=dict(title="nepisodes_solved"))
            self.win_repisodes_solved = self.master.vis.scatter(X=np.array(self.repisodes_solved_log), env=self.master.refs, win=self.win_repisodes_solved, opts=dict(title="repisodes_solved"))
        # logging
        self.master.logger.warning("Reporting       @ Step: " + str(eval_at_train_step) + " | Elapsed Time: " + str(time.time() - self.start_time))
        self.master.logger.warning("Iteration: {}; p_loss_avg: {}".format(eval_at_train_step, self.p_loss_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; v_loss_avg: {}".format(eval_at_train_step, self.v_loss_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; loss_avg: {}".format(eval_at_train_step, self.loss_avg_log[-1][1]))
        self.master._reset_training_loggings()
        self.master.logger.warning("Evaluating      @ Step: " + str(eval_at_train_step) + " | (" + str(eval_at_frame_step) + " frames)...")
        self.master.logger.warning("Evaluation        Took: " + str(time.time() - self.last_eval))
        self.master.logger.warning("Iteration: {}; entropy_avg: {}".format(eval_at_train_step, self.entropy_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; v_avg: {}".format(eval_at_train_step, self.v_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; steps_avg: {}".format(eval_at_train_step, self.steps_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; steps_std: {}".format(eval_at_train_step, self.steps_std_log[-1][1]))
        self.master.logger.warning("Iteration: {}; reward_avg: {}".format(eval_at_train_step, self.reward_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; reward_std: {}".format(eval_at_train_step, self.reward_std_log[-1][1]))
        self.master.logger.warning("Iteration: {}; nepisodes: {}".format(eval_at_train_step, self.nepisodes_log[-1][1]))
        self.master.logger.warning("Iteration: {}; nepisodes_solved: {}".format(eval_at_train_step, self.nepisodes_solved_log[-1][1]))
        self.master.logger.warning("Iteration: {}; repisodes_solved: {}".format(eval_at_train_step, self.repisodes_solved_log[-1][1]))
        self.last_eval = time.time()

        # save model
        self.master._save_model(eval_at_train_step, self.reward_avg_log[-1][1])

    def run(self):
        while self.master.train_step.value < self.master.steps:
            if time.time() - self.last_eval > self.master.eval_freq:
                self._eval_model()
        # we also do a final evaluation after training is done
        self._eval_model()

class ACERTester(ACERSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> ACER-Tester {Env & Model}")
        super(ACERTester, self).__init__(master, process_id)

        self.training = False   # choose actions w/ max probability
        self.model.train(self.training)
        self._reset_loggings()

        self.start_time = time.time()

    def _reset_loggings(self):
        # testing stats
        self.steps_avg_log = []
        self.steps_std_log = []
        self.reward_avg_log = []
        self.reward_std_log = []
        self.nepisodes_log = []
        self.nepisodes_solved_log = []
        self.repisodes_solved_log = []
        # placeholders for windows for online curve plotting
        if self.master.visualize:
            # evaluation stats
            self.win_steps_avg = "win_steps_avg"
            self.win_steps_std = "win_steps_std"
            self.win_reward_avg = "win_reward_avg"
            self.win_reward_std = "win_reward_std"
            self.win_nepisodes = "win_nepisodes"
            self.win_nepisodes_solved = "win_nepisodes_solved"
            self.win_repisodes_solved = "win_repisodes_solved"

    def run(self):
        test_step = 0
        test_nepisodes = 0
        test_nepisodes_solved = 0
        test_episode_steps = None
        test_episode_steps_log = []
        test_episode_reward = None
        test_episode_reward_log = []
        test_should_start_new = True
        while test_nepisodes < self.master.test_nepisodes:
            # TODO:
            test_nepisodes += 1

        self.steps_avg_log.append([test_nepisodes, np.mean(np.asarray(test_episode_steps_log))])
        self.steps_std_log.append([test_nepisodes, np.std(np.asarray(test_episode_steps_log))]); del test_episode_steps_log
        self.reward_avg_log.append([test_nepisodes, np.mean(np.asarray(test_episode_reward_log))])
        self.reward_std_log.append([test_nepisodes, np.std(np.asarray(test_episode_reward_log))]); del test_episode_reward_log
        self.nepisodes_log.append([test_nepisodes, test_nepisodes])
        self.nepisodes_solved_log.append([test_nepisodes, test_nepisodes_solved])
        self.repisodes_solved_log.append([test_nepisodes, (test_nepisodes_solved/test_nepisodes) if test_nepisodes > 0 else 0.])
        # plotting
        if self.master.visualize:
            self.win_steps_avg = self.master.vis.scatter(X=np.array(self.steps_avg_log), env=self.master.refs, win=self.win_steps_avg, opts=dict(title="steps_avg"))
            # self.win_steps_std = self.master.vis.scatter(X=np.array(self.steps_std_log), env=self.master.refs, win=self.win_steps_std, opts=dict(title="steps_std"))
            self.win_reward_avg = self.master.vis.scatter(X=np.array(self.reward_avg_log), env=self.master.refs, win=self.win_reward_avg, opts=dict(title="reward_avg"))
            # self.win_reward_std = self.master.vis.scatter(X=np.array(self.reward_std_log), env=self.master.refs, win=self.win_reward_std, opts=dict(title="reward_std"))
            self.win_nepisodes = self.master.vis.scatter(X=np.array(self.nepisodes_log), env=self.master.refs, win=self.win_nepisodes, opts=dict(title="nepisodes"))
            self.win_nepisodes_solved = self.master.vis.scatter(X=np.array(self.nepisodes_solved_log), env=self.master.refs, win=self.win_nepisodes_solved, opts=dict(title="nepisodes_solved"))
            self.win_repisodes_solved = self.master.vis.scatter(X=np.array(self.repisodes_solved_log), env=self.master.refs, win=self.win_repisodes_solved, opts=dict(title="repisodes_solved"))
        # logging
        self.master.logger.warning("Testing  Took: " + str(time.time() - self.start_time))
        self.master.logger.warning("Testing: steps_avg: {}".format(self.steps_avg_log[-1][1]))
        self.master.logger.warning("Testing: steps_std: {}".format(self.steps_std_log[-1][1]))
        self.master.logger.warning("Testing: reward_avg: {}".format(self.reward_avg_log[-1][1]))
        self.master.logger.warning("Testing: reward_std: {}".format(self.reward_std_log[-1][1]))
        self.master.logger.warning("Testing: nepisodes: {}".format(self.nepisodes_log[-1][1]))
        self.master.logger.warning("Testing: nepisodes_solved: {}".format(self.nepisodes_solved_log[-1][1]))
        self.master.logger.warning("Testing: repisodes_solved: {}".format(self.repisodes_solved_log[-1][1]))
