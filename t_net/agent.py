from collections import deque
from copy import deepcopy
from .fc_net import FCNet
import numpy as np
import torch
import random


class AgentSC(object):
    '''
    A classic deep Q Learning Agent for simplified tetris 99 environment. Implements DQN with fixed target and experience
    replay learning strategy
    Args:
        env (T99SC):                    environment to train on
        discount (float):               how important is the future rewards compared to the immediate ones [0,1]
        net (nn.module):                NN architecture and weights that will be copied to both target and policy nets
        learning rate (float):          speed of optimization
                                        internally
        criterion (nn.module):          loss function
        device (object):                device that will handle NN computations
        features (object):              function used to extract NN-acceptable features form env.state
        epsilon (float):                exploration (probability of random values given) value at the start
        mem_size (int):                 size of the cyclic buffer that records actions
    '''

    def __init__(self, env, discount, net, learning_rate, criterion, device, features, exploration_rate=0.1, mem_size=10000):

        # memory is a cyclic buffer
        self.memory = deque(maxlen=mem_size)
        # tetris environment
        self.env = env
        # gamma
        self.discount = discount
        # specifies a threshold, after which the amount of collected data is sufficient for starting training
        self.replay_start_size = 2000
        # set up nets
        self.policy_net = deepcopy(net)
        self.target_net = deepcopy(net)
        self.optimizer = torch.optim.Adam(self.target_net.parameters(), lr=learning_rate)
        self.criterion = criterion
        # choose a function to extract features
        self.get_features = features
        # set up strategy hyper parameters
        self.exploration_rate = exploration_rate

        # record the device for future use
        self.device = device
        # send the nets to the device
        self.policy_net.to(device)
        self.target_net.to(device)

    def add_to_memory(self, current_state, next_state, reward, done):
        # Adds a play to the replay memory buffer
        self.memory.append((current_state, next_state, reward, done))

    def predict_value(self, state):
        # Predicts the score for a certain state
        # remember that NN always returns a vector, even if its size == 1. So prediction actually lies in FCNet(inp)[0]
        return self.model(state)[0]

    def act(self):
        '''
        Makes a single-step update in accordance with epsilon-greedy strategy
        '''
        # observe the options we have for reward and next states of the player controlled by the agent
        options, rewards = self.env._observe(0)
        # check if we are exploring on this step
        if np.random.random_sample() <= self.exploration_rate:
            # if so, choose an action on random
            index = np.random.randint(0, high=len(rewards))
        # if we are exploiting on this step
        else:
            # get features for all next states
            feats = []
            for i in range(len(rewards)):
                feats.append(torch.from_numpy(self.get_features(options[i])))
            # then stack all possible net states into one tensor and send it to the GPU
            next_states = torch.stack(feats).type(torch.FloatTensor).to(self.device)
            # calculate predictions on the whole stack using target net (see algorithm)
            predictions = self.target_net(next_states)[:, 0]
            # choose greedy action
            index = torch.argmax(predictions).item()
        # now make a step according with a selected action, and record the reward
        action = {
            "reward": rewards[index],
            "state": options[index]
        }
        _, reward, done, _ = self.env.step(action)

        return reward, done

    def train(self, batch_size=128, update_freq=2000, steps=10000):
        '''
        Trains the agent by following DQN-learning algorithm
        '''
        # we will save the following statistics about the data
        cumulative_rewards = [0]
        steps_per_epoch = [0]
        # initialize epoch
        epoch = 0

        # repeats the algorithm steps times
        for i in range(steps):
            # record the step
            if i % 1000 == 0: print("calculating step", i)
            # get features for the current state
            current_state_features = self.get_features(self.env.state)
            # make an action, record the reward and check whether environment is done
            reward, done = self.act()
            # get next state features
            next_state_features = self.get_features(self.env.state)
            # record the current state, reward, next state, done in the cyclic buffer for future replay
            self.add_to_memory(current_state_features, reward, next_state_features, done)
            # record cumulative reward and steps done in the current epoch
            cumulative_rewards[epoch] += reward
            steps_per_epoch[epoch] += 1
            # if the environment is done, reboot it and reset counters
            if done:
                self.env.reset()
                epoch += 1
                cumulative_rewards.append(0)
                steps_per_epoch.append(0)

            # check if there is enough data to start training
            if len(self.memory) > self.replay_start_size:
                # sample a batch of transitions
                batch = random.sample(self.memory, batch_size)
                # init arrays to hold states and rewards
                batch_states = []
                batch_rewards = []
                batch_next_states = []
                # store features in arrays
                for j in range(len(batch)):
                    batch_states.append(torch.from_numpy(batch[j][0]))
                    batch_rewards.append(torch.tensor(batch[j][1]))
                    batch_next_states.append(torch.from_numpy(batch[j][2]))
                # stack tensors and send them to GPU
                torch_states = torch.stack(batch_states).type(torch.FloatTensor).to(self.device)
                torch_rewards = torch.stack(batch_rewards).type(torch.FloatTensor).to(self.device)
                torch_next_states = torch.stack(batch_next_states).type(torch.FloatTensor).to(self.device)
                # get the expected score for the next state using policy net
                q_next = self.policy_net(torch_next_states)[:, 0]
                # calculate target
                y_i = q_next + torch.tensor(self.discount) * torch_rewards
                # get the expected score for the current state using target net
                q_current = self.target_net(torch_states)[:, 0]

                # Fit the model to the given values
                loss = self.criterion(y_i, q_current)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # if it's time to update policy net
            if i % update_freq == 0:
                # pop up a message
                print("policy net updated")
                # update weights
                self.policy_net.load_state_dict(self.target_net.state_dict())

        return cumulative_rewards, steps_per_epoch

    #Saving functionality
        #Not working or tested, because there is no self.model
        #But should be good starting point, commented out to not mess with anything for now
        
    # def save_state(self,path="models/agent-sc-model.pth"):
        
    #     torch.save({
    #         'model_state_dict': self.model.state_dict(), 
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'discount' : self.discount,
    #         'replay_start_size' : self.replay_start_size,
    #         'exploration_rate' : self.exploration_rate,
    #         'mem_size' : self.mem_size,
    #         }, path)

    # def resume_state(self,path):
    #     checkpoint = torch.load(path)
    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.discount = checkpoint['discount']
    #     self.replay_start_size = checkpoint['replay_start_size']
    #     self.exploration_rate = checkpoint['exploration_rate']
    #     self.memory = deque(maxlen=checkpoint['mem_size'])

    #     #Don't forget to do .eval() or .train() now!