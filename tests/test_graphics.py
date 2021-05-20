import unittest
import os
import sys
import time

# list directories where packages are stored
# note that the parent directory of te repo is added automatically
GYM_FOLDER = "gym-t99"

# get this notebook's current working directory
nb_cwd = os.getcwd()
# get name of its parent directory
nb_parent = os.path.dirname(nb_cwd)
# add packages to path
sys.path.insert(len(sys.path), nb_parent)
sys.path.insert(len(sys.path), os.path.join(nb_parent, GYM_FOLDER))

import gym
registered = gym.envs.registration.registry.env_specs.copy()

from time import sleep
import gym_t99
from gym_t99.envs.state import Piece
import t_net
import numpy as np


class TestSuite(unittest.TestCase):
              
    def setUp(self):
        self.custom_gym = gym.make('gym_t99:t99-v0', num_players = 99, enemy="rand")
    
    def tearDown(self):
        pass
    
    # def test_basic_debug_output(self):
    #     frame = self.custom_gym.render(mode="debug")
    #     print(frame[0])

    # def test_window_show(self):
    #     #Window should show
    #     self.custom_gym.render(mode='human',show_window=True)
    #     sleep(4)
    #     self.custom_gym.close()
    #     sleep(4)
        
    # def test_window_no_show(self):
    #     #Window should not show
    #     self.custom_gym.render(mode='human',show_window=False)
    #     sleep(4)
    #     self.custom_gym.close()
    
    # def test_window_open_and_close(self):
    #     #Window should show
    #     self.custom_gym.render(mode='human',show_window=True)
    #     sleep(2)
    #     #Window should close, should show debug mode
    #     frame = self.custom_gym.render(mode='debug')
    #     sleep(4)
    #     self.custom_gym.render(mode='human',show_window=True)
    #     #Window should show again
    #     sleep(2)
    #     self.custom_gym.close()
    #     #Does not seem to work

    # def test_draw_pieces_small(self):

    #     self.custom_gym.state.players[0].board = \
    #                               np.array([[10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10,10,10,10,10,10,10,10,10,10,10,10,10]])
        
    #     self.custom_gym.state.players[1].board = \
    #                               np.array([[10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 6, 6, 0, 0, 0, 5, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 6, 0, 0, 5, 5, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10,10,10,10,10,10,10,10,10,10,10,10,10]])
        
    #     self.custom_gym.state.players[2].board = \
    #                               np.array([[10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 2, 0, 6, 6, 0, 0, 0, 5, 0, 0,10,10],
    #                                         [10,10, 2, 0, 0, 6, 0, 0, 5, 5, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0,10,10],
    #                                         [10,10,10,10,10,10,10,10,10,10,10,10,10,10]])
            
    #     self.custom_gym.state.players[3].board = \
    #                             np.array([[10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 2, 0, 6, 6, 0, 0, 0, 5, 0, 0,10,10],
    #                                     [10,10, 2, 0, 0, 6, 0, 4, 5, 5, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 0, 4, 4, 5, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0,10,10],
    #                                     [10,10,10,10,10,10,10,10,10,10,10,10,10,10]])
                
    #     self.custom_gym.state.players[4].board = \
    #                     np.array([[10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 2, 0, 6, 6, 0, 0, 0, 5, 0, 0,10,10],
    #                             [10,10, 2, 0, 0, 6, 0, 4, 5, 5, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 0, 4, 4, 5, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0,10,10],
    #                             [10,10, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0,10,10],
    #                             [10,10,10,10,10,10,10,10,10,10,10,10,10,10]])
       
      
       
    #     self.custom_gym.render(mode='human',show_window=True)
    #     time.sleep(15)

    def test_draw_pieces(self):
        self.custom_gym.state.players[2].place = 39
        self.custom_gym.state.players[4].place = 7
        self.custom_gym.state.players[0].board = \
                                  np.array([[10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 2, 0, 6, 6, 0, 0, 0, 5, 0, 0,10,10],
                                        [10,10, 2, 0, 0, 6, 0, 4, 5, 5, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 4, 4, 5, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 7, 7,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 7, 7,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10]])
        
        self.custom_gym.state.players[1].board = \
                                  np.array([[10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 2, 0, 6, 6, 0, 0, 0, 5, 0, 0,10,10],
                                        [10,10, 2, 0, 0, 6, 0, 4, 5, 5, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 4, 4, 5, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 7, 7,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 7, 7,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10]])
        
        self.custom_gym.state.players[2].board = \
                                  np.array([[10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 2, 0, 6, 6, 0, 0, 0, 5, 0, 0,10,10],
                                        [10,10, 2, 0, 0, 6, 0, 4, 5, 5, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 4, 4, 5, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 7, 7,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 7, 7,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10]])
            
        self.custom_gym.state.players[3].board = \
                                np.array([[10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 2, 0, 6, 6, 0, 0, 0, 5, 0, 0,10,10],
                                        [10,10, 2, 0, 0, 6, 0, 4, 5, 5, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 4, 4, 5, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 7, 7,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 7, 7,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10]])
                
        self.custom_gym.state.players[4].board = \
                        np.array([[10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 2, 0, 6, 6, 0, 0, 0, 5, 0, 0,10,10],
                                        [10,10, 2, 0, 0, 6, 0, 4, 5, 5, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 4, 4, 5, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 7, 7,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 7, 7,10,10],
                                        [10,10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,10,10],
                                        [10,10, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                                        [10,10,10,10,10,10,10,10,10,10,10,10,10,10]])
       
      
       
        self.custom_gym.render(mode='human',show_window=False)
        time.sleep(2)


if __name__ == '__main__':
    unittest.main()
