from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import time
import torch
from torch.autograd import Variable

from optims.helpers import adjust_learning_rate
from core.agent import Agent

class DQNAgent(Agent):
    def __init__(self, args, env_prototype, model_prototype, memory_prototype):
        super(DQNAgent, self).__init__(args, env_prototype, model_prototype, memory_prototype)
        self.logger.warning("<===================================> DQN")

        # env
        self.env = self.env_prototype(self.env_params)
        self.state_shape = self.env.state_shape
        self.action_dim  = self.env.action_dim

        # model
        self.model_params.state_shape = self.state_shape
        self.model_params.action_dim = self.action_dim
        self.model = self.model_prototype(self.model_params)
        self._load_model(self.model_file)   # load pretrained model if provided
        # target_model
        self.target_model = self.model_prototype(self.model_params)
        self._update_target_model_hard()

        # memory
        # NOTE: we instantiate memory objects only inside fit_model/test_model
        # NOTE: since in fit_model we need both replay memory and recent memory
        # NOTE: while in test_model we only need recent memory, in which case memory_size=0
        self.memory_params = args.memory_params

        # experience & states
        self._reset_states()

    def _reset_training_loggings(self):
        self._reset_testing_loggings()
        # during the evaluation in training, we additionally log for
        # the predicted Q-values and TD-errors on validation data
        self.v_avg_log = []
        self.tderr_avg_log = []
        # placeholders for windows for online curve plotting
        if self.visualize:
            self.win_v_avg = "win_v_avg"
            self.win_tderr_avg = "win_tderr_avg"

    def _reset_testing_loggings(self):
        # setup logging for testing/evaluation stats
        self.steps_avg_log = []
        self.steps_std_log = []
        self.reward_avg_log = []
        self.reward_std_log = []
        self.nepisodes_log = []
        self.nepisodes_solved_log = []
        self.repisodes_solved_log = []
        # placeholders for windows for online curve plotting
        if self.visualize:
            self.win_steps_avg = "win_steps_avg"
            self.win_steps_std = "win_steps_std"
            self.win_reward_avg = "win_reward_avg"
            self.win_reward_std = "win_reward_std"
            self.win_nepisodes = "win_nepisodes"
            self.win_nepisodes_solved = "win_nepisodes_solved"
            self.win_repisodes_solved = "win_repisodes_solved"

    def _reset_states(self):
        self._reset_experience()
        self.recent_action = None
        self.recent_observation = None

    # Hard update every `target_model_update` steps.
    def _update_target_model_hard(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
    def _update_target_model_soft(self):
        for i, (key, target_weights) in enumerate(self.target_model.state_dict().iteritems()):
            target_weights += self.target_model_update * self.model.state_dict()[key]

    def _sample_validation_data(self):
        self.logger.warning("Validation Data @ Step: " + str(self.step))
        self.valid_data = self.memory.sample(self.valid_size)

    def _compute_validation_stats(self):
        return self._get_q_update(self.valid_data)

    def _get_q_update(self, experiences): # compute temporal difference error for a batch
        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch_vb    = Variable(torch.from_numpy(np.array(tuple(experiences[i].state0 for i in range(len(experiences))))).type(self.dtype))
        action_batch_vb    = Variable(torch.from_numpy(np.array(tuple(experiences[i].action for i in range(len(experiences))))).long())
        reward_batch_vb    = Variable(torch.from_numpy(np.array(tuple(experiences[i].reward for i in range(len(experiences)))))).type(self.dtype)
        state1_batch_vb    = Variable(torch.from_numpy(np.array(tuple(experiences[i].state1 for i in range(len(experiences))))).type(self.dtype))
        terminal1_batch_vb = Variable(torch.from_numpy(np.array(tuple(0. if experiences[i].terminal1 else 1. for i in range(len(experiences)))))).type(self.dtype)

        if self.use_cuda:
            action_batch_vb = action_batch_vb.cuda()

        # Compute target Q values for mini-batch update.
        if self.enable_double_dqn:
            # According to the paper "Deep Reinforcement Learning with Double Q-learning"
            # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
            # while the target network is used to estimate the Q value.
            q_values_vb = self.model(state1_batch_vb)
            # Detach this variable from the current graph since we don't want gradients to propagate
            q_values_vb = Variable(q_values_vb.data)
            # _, q_max_actions_vb = q_values_vb.max(dim=1)              # 0.1.12
            _, q_max_actions_vb = q_values_vb.max(dim=1, keepdim=True)  # 0.2.0
            # Now, estimate Q values using the target network but select the values with the
            # highest Q value wrt to the online model (as computed above).
            next_max_q_values_vb = self.target_model(state1_batch_vb)
            # Detach this variable from the current graph since we don't want gradients to propagate
            next_max_q_values_vb = Variable(next_max_q_values_vb.data)
            next_max_q_values_vb = next_max_q_values_vb.gather(1, q_max_actions_vb)
        else:
            # Compute the q_values given state1, and extract the maximum for each sample in the batch.
            # We perform this prediction on the target_model instead of the model for reasons
            # outlined in Mnih (2015). In short: it makes the algorithm more stable.
            next_max_q_values_vb = self.target_model(state1_batch_vb)
            # Detach this variable from the current graph since we don't want gradients to propagate
            next_max_q_values_vb = Variable(next_max_q_values_vb.data)
            # next_max_q_values_vb, _ = next_max_q_values_vb.max(dim = 1)               # 0.1.12
            next_max_q_values_vb, _ = next_max_q_values_vb.max(dim = 1, keepdim=True)   # 0.2.0

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the targets accordingly
        # but only for the affected output units (as given by action_batch).
        current_q_values_vb = self.model(state0_batch_vb).gather(1, action_batch_vb.unsqueeze(1)).squeeze()
        # Set discounted reward to zero for all states that were terminal.
        next_max_q_values_vb = next_max_q_values_vb * terminal1_batch_vb.unsqueeze(1)
        # expected_q_values_vb = reward_batch_vb + self.gamma * next_max_q_values_vb            # 0.1.12
        expected_q_values_vb = reward_batch_vb + self.gamma * next_max_q_values_vb.squeeze()    # 0.2.0
        # Compute temporal difference error, use huber loss to mitigate outlier impact
        # TODO: can optionally use huber loss from here: https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
        td_error_vb = self.value_criteria(current_q_values_vb, expected_q_values_vb)

        # return v_avg, tderr_avg_vb
        if not self.training:   # then is being called from _compute_validation_stats, which is just doing inference
            td_error_vb = Variable(td_error_vb.data) # detach it from the graph
        return next_max_q_values_vb.data.clone().mean(), td_error_vb

    def _epsilon_greedy(self, q_values_ts):
        # calculate epsilon
        if self.training:   # linearly anneal epsilon
            self.eps = self.eps_end + max(0, (self.eps_start - self.eps_end) * (self.eps_decay - max(0, self.step - self.learn_start)) / self.eps_decay)
        else:
            self.eps = self.eps_eval
        # choose action
        if np.random.uniform() < self.eps:  # then we choose a random action
            action = random.randrange(self.action_dim)
        else:                               # then we choose the greedy action
            if self.use_cuda:
                action = np.argmax(q_values_ts.cpu().numpy())
            else:
                action = np.argmax(q_values_ts.numpy())
        return action

    def _forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        state_ts = torch.from_numpy(np.array(state)).unsqueeze(0).type(self.dtype)
        q_values_ts = self.model(Variable(state_ts, volatile=True)).data # NOTE: only doing inference here, so volatile=True
        if self.training and self.step < self.learn_start:  # then we don't do any learning, just accumulate experiences into replay memory
            action = random.randrange(self.action_dim)      # thus we only randomly sample actions here, since the model hasn't been updated at all till now
        else:
            action = self._epsilon_greedy(q_values_ts)

        # Book keeping
        self.recent_observation = observation
        self.recent_action = action

        return action

    def _backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            # NOTE: so the tuples stored in memory corresponds to:
            # NOTE: in recent_observation(state0), take recent_action(action), get reward(reward), ends up in terminal(terminal1)
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training = self.training)

        if not self.training:
            # We're done here. No need to update the replay memory since we only use the
            # recent memory to obtain the state over the most recent observations.
            return

        # sample validation data right before training started
        # NOTE: here validation data is not entirely clean since the agent might see those data during training
        # NOTE: but it's ok as is also the case in the original dqn code, cos those data are not used to judge performance like in supervised learning
        # NOTE: but rather to inspect the whole learning procedure; of course we can separate those entirely from the training data but it's not worth the effort
        if self.step == self.learn_start + 1:
            self._sample_validation_data()
            self.logger.warning("Start  Training @ Step: " + str(self.step))

        # Train the network on a single stochastic batch.
        if self.step > self.learn_start and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            # Compute temporal difference error
            _, td_error_vb = self._get_q_update(experiences)
            # Construct optimizer and clear old gradients
            # TODO: can linearly anneal the lr here thus we would have to create a new optimizer here
            # TODO: we leave the lr constant here for now and wait for update threads maybe from: https://discuss.pytorch.org/t/adaptive-learning-rate/320/11
            self.optimizer.zero_grad()
            # run backward pass and clip gradient
            td_error_vb.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-self.clip_grad, self.clip_grad)
            # Perform the update
            self.optimizer.step()

        # adjust learning rate if enabled
        if self.lr_decay:
            self.lr_adjusted = max(self.lr * (self.steps - self.step) / self.steps, 1e-32)
            adjust_learning_rate(self.optimizer, self.lr_adjusted)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self._update_target_model_hard()    # Hard update every `target_model_update` steps.
        if self.target_model_update < 1.:       # TODO: have not tested
            self._update_target_model_soft()    # Soft update with `(1 - target_model_update) * old + target_model_update * new`.

        return

    def fit_model(self):
        # memory
        self.memory = self.memory_prototype(limit = self.memory_params.memory_size,
                                            window_length = self.memory_params.hist_len)
        self.eps = self.eps_start
        # self.optimizer = self.optim(self.model.parameters(), lr=self.lr, alpha=0.95, eps=0.01, weight_decay=self.weight_decay)  # RMSprop
        self.optimizer = self.optim(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)    # Adam
        self.lr_adjusted = self.lr

        self.logger.warning("<===================================> Training ...")
        self.training = True
        self._reset_training_loggings()

        self.start_time = time.time()
        self.step = 0

        nepisodes = 0
        nepisodes_solved = 0
        episode_steps = None
        episode_reward = None
        total_reward = 0.
        should_start_new = True
        while self.step < self.steps:
            if should_start_new:    # start of a new episode
                episode_steps = 0
                episode_reward = 0.
                # Obtain the initial observation by resetting the environment
                self._reset_states()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                if not self.training:
                    if self.visualize: self.env.visual()
                    if self.render: self.env.render()
                # reset flag
                should_start_new = False
            # Run a single step
            # This is where all of the work happens. We first perceive and compute the action
            # (forward step) and then use the reward to improve (backward step)
            action = self._forward(self.experience.state1)
            reward = 0.
            for _ in range(self.action_repetition):
                self.experience = self.env.step(action)
                if not self.training:
                    if self.visualize: self.env.visual()
                    if self.render: self.env.render()
                reward += self.experience.reward
                if self.experience.terminal1:
                    should_start_new = True
                    break
            if self.early_stop and (episode_steps + 1) >= self.early_stop or (self.step + 1) % self.eval_freq == 0:
                # to make sure the historic observations for the first hist_len-1 steps in (the next episode / eval) would be clean
                should_start_new = True
            if should_start_new:
                self._backward(reward, True)
            else:
                self._backward(reward, self.experience.terminal1)

            episode_steps += 1
            episode_reward += reward
            self.step += 1

            if should_start_new:
                # We are in a terminal state but the agent hasn't yet seen it. We therefore
                # perform one more forward-backward call and simply ignore the action before
                # resetting the environment. We need to pass in "terminal=False" here since
                # the *next* state, that is the state of the newly reset environment, is
                # always non-terminal by convention.
                self._forward(self.experience.state1)   # recent_observation & recent_action get updated
                self._backward(0., False)               # recent experience gets pushed into memory
                                                        # NOTE: the append happened inside here is just trying to save s1, none of a,r,t are used for this terminal s1 when sample
                total_reward += episode_reward
                nepisodes += 1
                if self.experience.terminal1:
                    nepisodes_solved += 1
                self._reset_states()
                episode_steps = None
                episode_reward = None

            # report training stats
            if self.step % self.prog_freq == 0:
                self.logger.warning("Reporting       @ Step: " + str(self.step) + " | Elapsed Time: " + str(time.time() - self.start_time))
                self.logger.warning("Training Stats:   lr:               {}".format(self.lr_adjusted))
                self.logger.warning("Training Stats:   epsilon:          {}".format(self.eps))
                self.logger.warning("Training Stats:   total_reward:     {}".format(total_reward))
                self.logger.warning("Training Stats:   avg_reward:       {}".format(total_reward/nepisodes if nepisodes > 0 else 0.))
                self.logger.warning("Training Stats:   nepisodes:        {}".format(nepisodes))
                self.logger.warning("Training Stats:   nepisodes_solved: {}".format(nepisodes_solved))
                self.logger.warning("Training Stats:   repisodes_solved: {}".format(nepisodes_solved/nepisodes if nepisodes > 0 else 0.))

            # evaluation & checkpointing
            if self.step > self.learn_start and self.step % self.eval_freq == 0:
                # Set states for evaluation
                self.training = False
                self.logger.warning("Evaluating      @ Step: " + str(self.step))
                self._eval_model()

                # Set states for resume training
                self.training = True
                self.logger.warning("Resume Training @ Step: " + str(self.step))
                should_start_new = True

    def _eval_model(self):
        self.training = False
        eval_step = 0

        eval_nepisodes = 0
        eval_nepisodes_solved = 0
        eval_episode_steps = None
        eval_episode_steps_log = []
        eval_episode_reward = None
        eval_episode_reward_log = []
        eval_should_start_new = True
        while eval_step < self.eval_steps:
            if eval_should_start_new:   # start of a new episode
                eval_episode_steps = 0
                eval_episode_reward = 0.
                # Obtain the initial observation by resetting the environment
                self._reset_states()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                if not self.training:
                    if self.visualize: self.env.visual()
                    if self.render: self.env.render()
                # reset flag
                eval_should_start_new = False
            # Run a single step
            eval_action = self._forward(self.experience.state1)
            eval_reward = 0.
            for _ in range(self.action_repetition):
                self.experience = self.env.step(eval_action)
                if not self.training:
                    if self.visualize: self.env.visual()
                    if self.render: self.env.render()
                eval_reward += self.experience.reward
                if self.experience.terminal1:
                    eval_should_start_new = True
                    break
            if self.early_stop and (eval_episode_steps + 1) >= self.early_stop or (eval_step + 1) == self.eval_steps:
                # to make sure the historic observations for the first hist_len-1 steps in (the next episode / resume training) would be clean
                eval_should_start_new = True
            # NOTE: here NOT doing backprop, only adding into recent memory
            if eval_should_start_new:
                self._backward(eval_reward, True)
            else:
                self._backward(eval_reward, self.experience.terminal1)

            eval_episode_steps += 1
            eval_episode_reward += eval_reward
            eval_step += 1

            if eval_should_start_new:
                # We are in a terminal state but the agent hasn't yet seen it. We therefore
                # perform one more forward-backward call and simply ignore the action before
                # resetting the environment. We need to pass in "terminal=False" here since
                # the *next* state, that is the state of the newly reset environment, is
                # always non-terminal by convention.
                self._forward(self.experience.state1) # recent_observation & recent_action get updated
                self._backward(0., False)             # NOTE: here NOT doing backprop, only adding into recent memory

                eval_nepisodes += 1
                if self.experience.terminal1:
                    eval_nepisodes_solved += 1

                # This episode is finished, report and reset
                eval_episode_steps_log.append([eval_episode_steps])
                eval_episode_reward_log.append([eval_episode_reward])
                self._reset_states()
                eval_episode_steps = None
                eval_episode_reward = None

        # Computing validation stats
        v_avg, tderr_avg_vb = self._compute_validation_stats()
        # Logging for this evaluation phase
        self.v_avg_log.append([self.step, v_avg])
        self.tderr_avg_log.append([self.step, tderr_avg_vb.data.clone().mean()])
        self.steps_avg_log.append([self.step, np.mean(np.asarray(eval_episode_steps_log))])
        self.steps_std_log.append([self.step, np.std(np.asarray(eval_episode_steps_log))]); del eval_episode_steps_log
        self.reward_avg_log.append([self.step, np.mean(np.asarray(eval_episode_reward_log))])
        self.reward_std_log.append([self.step, np.std(np.asarray(eval_episode_reward_log))]); del eval_episode_reward_log
        self.nepisodes_log.append([self.step, eval_nepisodes])
        self.nepisodes_solved_log.append([self.step, eval_nepisodes_solved])
        self.repisodes_solved_log.append([self.step, (eval_nepisodes_solved/eval_nepisodes) if eval_nepisodes > 0 else 0])
        # plotting
        if self.visualize:
            self.win_v_avg = self.vis.scatter(X=np.array(self.v_avg_log), env=self.refs, win=self.win_v_avg, opts=dict(title="v_avg"))
            self.win_tderr_avg = self.vis.scatter(X=np.array(self.tderr_avg_log), env=self.refs, win=self.win_tderr_avg, opts=dict(title="tderr_avg"))
            self.win_steps_avg = self.vis.scatter(X=np.array(self.steps_avg_log), env=self.refs, win=self.win_steps_avg, opts=dict(title="steps_avg"))
            # self.win_steps_std = self.vis.scatter(X=np.array(self.steps_std_log), env=self.refs, win=self.win_steps_std, opts=dict(title="steps_std"))
            self.win_reward_avg = self.vis.scatter(X=np.array(self.reward_avg_log), env=self.refs, win=self.win_reward_avg, opts=dict(title="reward_avg"))
            # self.win_reward_std = self.vis.scatter(X=np.array(self.reward_std_log), env=self.refs, win=self.win_reward_std, opts=dict(title="reward_std"))
            self.win_nepisodes = self.vis.scatter(X=np.array(self.nepisodes_log), env=self.refs, win=self.win_nepisodes, opts=dict(title="nepisodes"))
            self.win_nepisodes_solved = self.vis.scatter(X=np.array(self.nepisodes_solved_log), env=self.refs, win=self.win_nepisodes_solved, opts=dict(title="nepisodes_solved"))
            self.win_repisodes_solved = self.vis.scatter(X=np.array(self.repisodes_solved_log), env=self.refs, win=self.win_repisodes_solved, opts=dict(title="repisodes_solved"))
        # logging
        self.logger.warning("Iteration: {}; v_avg: {}".format(self.step, self.v_avg_log[-1][1]))
        self.logger.warning("Iteration: {}; tderr_avg: {}".format(self.step, self.tderr_avg_log[-1][1]))
        self.logger.warning("Iteration: {}; steps_avg: {}".format(self.step, self.steps_avg_log[-1][1]))
        self.logger.warning("Iteration: {}; steps_std: {}".format(self.step, self.steps_std_log[-1][1]))
        self.logger.warning("Iteration: {}; reward_avg: {}".format(self.step, self.reward_avg_log[-1][1]))
        self.logger.warning("Iteration: {}; reward_std: {}".format(self.step, self.reward_std_log[-1][1]))
        self.logger.warning("Iteration: {}; nepisodes: {}".format(self.step, self.nepisodes_log[-1][1]))
        self.logger.warning("Iteration: {}; nepisodes_solved: {}".format(self.step, self.nepisodes_solved_log[-1][1]))
        self.logger.warning("Iteration: {}; repisodes_solved: {}".format(self.step, self.repisodes_solved_log[-1][1]))

        # save model
        self._save_model(self.step, self.reward_avg_log[-1][1])

    def test_model(self):
        # memory    # NOTE: here we don't need a replay memory, just a recent memory
        self.memory = self.memory_prototype(limit = 0,
                                            window_length = self.memory_params.hist_len)
        self.eps = self.eps_eval

        self.logger.warning("<===================================> Testing ...")
        self.training = False
        self._reset_testing_loggings()

        self.start_time = time.time()
        self.step = 0

        test_nepisodes = 0
        test_nepisodes_solved = 0
        test_episode_steps = None
        test_episode_steps_log = []
        test_episode_reward = None
        test_episode_reward_log = []
        test_should_start_new = True
        while test_nepisodes < self.test_nepisodes:
            if test_should_start_new:   # start of a new episode
                test_episode_steps = 0
                test_episode_reward = 0.
                # Obtain the initial observation by resetting the environment
                self._reset_states()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                if not self.training:
                    if self.visualize: self.env.visual()
                    if self.render: self.env.render()
                # reset flag
                test_should_start_new = False
            # Run a single step
            test_action = self._forward(self.experience.state1)
            test_reward = 0.
            for _ in range(self.action_repetition):
                self.experience = self.env.step(test_action)
                if not self.training:
                    if self.visualize: self.env.visual()
                    if self.render: self.env.render()
                test_reward += self.experience.reward
                if self.experience.terminal1:
                    test_should_start_new = True
                    break
            if self.early_stop and (test_episode_steps + 1) >= self.early_stop:
                # to make sure the historic observations for the first hist_len-1 steps in (the next episode / resume training) would be clean
                test_should_start_new = True
            # NOTE: here NOT doing backprop, only adding into recent memory
            if test_should_start_new:
                self._backward(test_reward, True)
            else:
                self._backward(test_reward, self.experience.terminal1)

            test_episode_steps += 1
            test_episode_reward += test_reward
            self.step += 1

            if test_should_start_new:
                # We are in a terminal state but the agent hasn't yet seen it. We therefore
                # perform one more forward-backward call and simply ignore the action before
                # resetting the environment. We need to pass in "terminal=False" here since
                # the *next* state, that is the state of the newly reset environment, is
                # always non-terminal by convention.
                self._forward(self.experience.state1) # recent_observation & recent_action get updated
                self._backward(0., False)             # NOTE: here NOT doing backprop, only adding into recent memory

                test_nepisodes += 1
                if self.experience.terminal1:
                    test_nepisodes_solved += 1

                # This episode is finished, report and reset
                test_episode_steps_log.append([test_episode_steps])
                test_episode_reward_log.append([test_episode_reward])
                self._reset_states()
                test_episode_steps = None
                test_episode_reward = None

        # Logging for this testing phase
        self.steps_avg_log.append([self.step, np.mean(np.asarray(test_episode_steps_log))])
        self.steps_std_log.append([self.step, np.std(np.asarray(test_episode_steps_log))]); del test_episode_steps_log
        self.reward_avg_log.append([self.step, np.mean(np.asarray(test_episode_reward_log))])
        self.reward_std_log.append([self.step, np.std(np.asarray(test_episode_reward_log))]); del test_episode_reward_log
        self.nepisodes_log.append([self.step, test_nepisodes])
        self.nepisodes_solved_log.append([self.step, test_nepisodes_solved])
        self.repisodes_solved_log.append([self.step, (test_nepisodes_solved/test_nepisodes) if test_nepisodes > 0 else 0.])
        # plotting
        if self.visualize:
            self.win_steps_avg = self.vis.scatter(X=np.array(self.steps_avg_log), env=self.refs, win=self.win_steps_avg, opts=dict(title="steps_avg"))
            # self.win_steps_std = self.vis.scatter(X=np.array(self.steps_std_log), env=self.refs, win=self.win_steps_std, opts=dict(title="steps_std"))
            self.win_reward_avg = self.vis.scatter(X=np.array(self.reward_avg_log), env=self.refs, win=self.win_reward_avg, opts=dict(title="reward_avg"))
            # self.win_reward_std = self.vis.scatter(X=np.array(self.reward_std_log), env=self.refs, win=self.win_reward_std, opts=dict(title="reward_std"))
            self.win_nepisodes = self.vis.scatter(X=np.array(self.nepisodes_log), env=self.refs, win=self.win_nepisodes, opts=dict(title="nepisodes"))
            self.win_nepisodes_solved = self.vis.scatter(X=np.array(self.nepisodes_solved_log), env=self.refs, win=self.win_nepisodes_solved, opts=dict(title="nepisodes_solved"))
            self.win_repisodes_solved = self.vis.scatter(X=np.array(self.repisodes_solved_log), env=self.refs, win=self.win_repisodes_solved, opts=dict(title="repisodes_solved"))
        # logging
        self.logger.warning("Testing  Took: " + str(time.time() - self.start_time))
        self.logger.warning("Testing: steps_avg: {}".format(self.steps_avg_log[-1][1]))
        self.logger.warning("Testing: steps_std: {}".format(self.steps_std_log[-1][1]))
        self.logger.warning("Testing: reward_avg: {}".format(self.reward_avg_log[-1][1]))
        self.logger.warning("Testing: reward_std: {}".format(self.reward_std_log[-1][1]))
        self.logger.warning("Testing: nepisodes: {}".format(self.nepisodes_log[-1][1]))
        self.logger.warning("Testing: nepisodes_solved: {}".format(self.nepisodes_solved_log[-1][1]))
        self.logger.warning("Testing: repisodes_solved: {}".format(self.repisodes_solved_log[-1][1]))
