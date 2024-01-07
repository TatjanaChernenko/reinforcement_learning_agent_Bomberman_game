
# Requires Keras, Tensorflow!

from random import shuffle
from time import time, sleep

# Numpy
import numpy as np

# Random
import random as random

# Deque for memory
from collections import deque

# Tensorflow
import tensorflow as tf

# Keras
import keras.backend as K

from keras.models     import Sequential
from keras.models     import Model
from keras.models     import load_model

from keras.layers     import Dense
from keras.layers     import Input
from keras.layers     import Conv2D
from keras.layers     import Conv1D
from keras.layers     import Lambda
from keras.layers     import Flatten
from keras.layers     import GaussianDropout
from keras.layers     import Layer
from keras.layers     import add
from keras.layers     import Concatenate
from keras.layers     import Average

from keras.optimizers import Adam

from keras import activations
from keras import initializers

from keras.engine.base_layer import InputSpec

# Import settings
from settings import s
from settings import e

# OS
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#np.set_printoptions(threshold=np.nan) # If arrays should always be printed completely.

# SumTree has nan if called for the first time, ignore warning!
np.seterr(invalid='ignore')

""" General comments:
    > Create a competitive version in the end, with everything not necessary for using a trained model deleted for better performance.
      They will run it with learning False, so change remember etc. so that releasing from oscillation still works!!!!!
    > Everything below a dueling double convolutional network with PER doesn't seem to collect coins?
    > Huge boost from explicitly using symmetry!
    > USE CUDA 9!!! CUDA 10 apparently does not work with tensorflow 12?
    > Make own avatar and bomb!
    > np.load()[:n] seems to keep the whole np.load() in the memory! Avoid writing something like this!!
    > ON my CPU, predicting actions takes about 0.3-0.4 seconds!
      But the CPU they use should be stronger, so maybe 0.2 seconds?
"""

""" Possible Improvements:
    > Use policy gradients?
      See https://github.com/go2sea/A3C/blob/master/a3C.py
    > Try explicitly using symmetry for Dense NN? Just use Flatten after splitting Input, possibly with extra Lambda layer to delete useless d.o.f.
    > Saving the model does not work? Saving the weights does!
    > Try Noisy network again? Implementation basically right, but could be shorter! Define NoisyDense! Should sigma be adjusted because of averaging?
      See https://github.com/go2sea/NoisyNetDQN/blob/master/myUtils.py and https://github.com/wenh123/NoisyNet-DQN/blob/master/tf_util.py
      Better initialisation?
    > Can also get learning_bool from game_state, at least with new environment!
    > Clean up folders!
    > Use numba for speedup?! SumTree, Memory!
    > Save compressed SumTree after training? Just store the arrays of Memory!
    
    > Run in colab on TPU??
        > Change environment so that it doesnt run in subfolders. rather, change the call to folders? Works!!!
    
    > Multi Q learning?
    
    > Dont turn learning in main off! releasing from oscillation no longer works!!
      use self.learning_bool=False
    
    > Change to working directory, relevant for us?
    
    > How fast is predicting without GPU???
      aboute 6/7 ms with GPU, CPU half a second?
    
    > memory not always n MAX etc.
    
    > Whose bomb not possible since the environment does not save it (still possible), so set self.value_opp_elim = self.killed_opp ?
    
    > predict_action with first only works for single states, but this is irrelevant?
    
    > a few loops are still there , [ for ...], but they are necessary?
    
    > Fit impossible actions to wait??
    
    > elu for conv?
    
    > use mae mean absolute error for adam???!
    
    > scale plot with fitcount??
"""

""" Possible Environment Improvements:
    > Version of the game that can run on colab?
      No visual necessary, but for training purposes
      pygame does not work in a notebook?
    > Adjusting the settings from inside the agent, or at least the loop in main.
      Fairly easy to implement, just turn the relevant setting into a local variable (self._) and write a function that adjusts it!
      But would still be helpful if it already existed for every setting!
    > The last value for the bomb counter should be one instead of zero?
    > Whose bomb?
"""

""" How to use:
    > Turn self.generate_random_data to True to generate random data that will fill up the memory before starting. Run once with True,
      it will generate a file from it. Run with False afterwards, it will load from the file. Repeat this for new reward values and state versions.
    > Turn self.learning_bool to True if you want to train a new network, to False if you want to load a trained model. They are automatically saved
      in /model, model.h5 is always the newest version. Change self.load_model to load a different model (e.g. "modelN20").
    > Set self.usePER to True for prioritized experience replay from a SumTree, to False for random sampling from a deque.
    > There are different versions of create_state and build_model. The convolutional networks need stacks of pictures as input, the dense
      networks need an array (the collapsed pictures). Be sure to change state_size accordingly. Change the used model and state in setup by assigning
      self.model_version and self.state_version.
    > For self.use_target True it will use a model and a target model that is only updated every once in a while, for False it will only use one model.
    > Change self.action_size to 6 if all actions should be possible, to 5 if only BOMB should be excluded, to 4 for only walking. Change self.pactions
      accordingly.
    > Change the self.value_... to adjust the rewards for specific actions.
    > self.training_data_size or self.experience_size, depending on whether PER is used, determine the memory size. Both can safely flow over.
    > self.PER_e, self.PER_a, self.PER_b and self.PER_b_increment_per_sampling are parameters of PER. The increment determines how fast the sampling
      transitions to random, apparently this influences convergence.
    > self.learning_rate determines the rate with which our model learns.
    > self.gamma is the discount for future rewards.
    > self.epsilon, self.epsilon_min and self.epsilon_decay determine how fast the randomness in choosing the action decays. For using a noisy network
      set self.epsilon to zero, it will stay that way.
    > self.replay_batch_size, self.replay_batch_times determine how many steps are sampled and fitted at once, how often after every round.
    > The target model weights are updated every self.update_every rounds.
"""

""" Parameters:
"""

""" State:
    > Deleting constant values is probably harmful for the convolutional layers!
    > Collecting them in one 2D array is not really useful, the convolutional layers implicitly already use this.
"""


def setup(self):
    """Setup of the agent."""
    
    # Get a new seed, so that every run is unique.
    np.random.seed()
    
    simple_setup(self)
    
    
    ####### Parameters that need to be adjusted
    
    # Names
    self.name = "SimpleNobel"
        # Name of the agent
    self.save_as = "model"
        # Name of the model
    self.load_weights = "model_weights"
        # Name of the weights hat should be loaded
    self.load_memory = "memory"
        # Name of the memory that should be loaded
    self.store_random_as = "experience_random"
        # How the random experience should be stored
    
    self.save_perf = "versions/simpleagenttask3new"
        # Name of the version
    
    # Actions
    self.pactions = np.asarray([10]*4+[2,4]) # np.asarray([20]*4+[0,0])
        # Probabilities for certain events when acting randomly, they don't have to sum to one!
    self.force_random = False
        # Force random actions after static decision / oscillation
    self.random_actions_after_osc = 3
        # Amount of random actions taken after an oscillation is detected
    self.random_actions_after_wait = 3
        # Amount of random actions taken after useless waiting is detected
    self.compare_diamond_size = 4 # 3 only without enemies
        # 2*n+1 is the size of the diamond that is used to evaluate whether states are equal
    self.compare_diamond_bomb_size = 4
        # 2*n+1 is the size of the diamond that is used to evaluate whether there are bombs in the proximity of self
    self.no_waiting_bomb_range = False
        # Do not wait when in range of a bomb
    self.no_wait_on_bomb = False
        # Do not wait while standing on a bomb
    self.noimpossible = False
        # Do not try to execute impossible actions, also prohibits walking into explosions
    self.explosions_impossible = False
    self.explosions_impossible_random = False
        # Whether walking into explosions should be forbidden for the model and random actions
    self.first_not_bomb = False
        # First action usually should not be bomb
    self.possible_future_state = False
        # Use predicted future state to determine possible actions, i.e. do not run into explosions in the next step
        # This could be bad!?
    self.max_loop = 30
    self.min_loop = 8
    self.prevent_loops = False
    self.random_after_loop = 5
        # Max, min length of loops that should be checked
        # Takes about 1 ms to check till 30 for loops, worth it since it likes to get stuck!!!
    self.fit_impossible_as_wait = False
        # Fit impossible actions to the Q value wait would give
        # Really bad!!! fits walking into explosions to wait!!! But possibly useful if walking into explosions is allowed?
    self.abs_error_after_fit = False
        # Use absolute error after fitting, not before fitting, for PER
    
    # Size of state and action space
    self.state_size = (13,17,17)
        # "SMALL",          "ALL",          "ALLOH"         "ALLOHL"
        # (3,17,17)         (6,17,17)       (9,17,17)       (13,17,17)
    self.action_size = 6 # Normally len(self.actions)
    
    # Model, State
    self.model_version = "DUELCONVSYM"
    self.state_version = "ALLOHL"
    self.use_target = True
        # Use target model
    
    # Values for the reward function
    self.value_walk = 0 # Successful step
    self.value_wait = 0 # Waited
    self.value_impossible = -50 # Impossible action
    self.value_bomb_drop = 0 # Dropped bomb
    self.value_bomb_explode = 0 # Bomb exploded
    self.value_crate = 100 # Crate destroyed
    self.value_coin_found = 0 # Coin found
    self.value_coin = 1000 # Coin collected
    self.value_killed_opp = 5000 # Killed opponent
    self.value_killed_self = -1000 # Killed self
    self.value_got_killed = -2000 # Got killed
    self.value_opp_elim = 3000 # Opponent eliminated by someone else
    self.value_survived = 0 # Survived round
    
    self.value_steps = 0 # Number of steps, not sure whether this is useful?
    self.value_useless_waiting = -100 # Extra punishment for useless waiting
    self.value_osc = -100 # Punishment for oscillation between two states
    self.value_loop = -100 # Punishment for loop
    self.value_useless_bomb = -50 # Extra punishment for useless bomb
    self.value_wait_range_bomb = -100 # Extra punishment for waiting while in range of a bomb
    self.stuck_bomb = -100 # Punishment for getting stuck behind bomb, does not count explosions as impossible to step on!
    
    # Memory parameters
    self.training_data_size = 60000
         #Max size of the deque.
    self.experience_size = 60000
        # SumTree memory size.
    self.PER_max = 50. # 20
        # Has to be float! Maximum priority (to the power PER_a) for a leaf in our SumTree.
    self.PER_e = 0.1 # 0.1?
        # Lower bound for the priority (to the power PER_a), so elements never have probability 0.
    self.PER_a = 0.6 # 0.5
        # Parameter for the randomness of sampled data, a=0 pure random.
    self.PER_b = 0.4 # 0.4
        # How the fit gets adjusted to make up for the priority sampling, 1 is equivalent to deque?
    self.PER_b_increment = (1-self.PER_b)/140000 # /128/150
        # Every time a batch is sampled, b gets raised by this until b=1.
    self.save_memory = False
        # If memory should be periodically saved
    
    # Learning parameters
    self.generate_random_data = False
        # If True, completely random data is generated and stored at the end.
    self.learning_bool = False
        # If True normal, if False epsilon=epsilon_min and uses trained model
    self.learning_bool_again = False
        # If True load trained model to continue training
    self.usePER = False
        # If True remember/sample with PER, if False remember/sample with deque.
    self.load_memory = False
        # Load old memory
    self.gamma = 0.97 # 0.99
        # Discount rate for future rewards.
    self.learning_rate = 0 # 0.001
        # Learning rate for Adam
    self.learning_rate_decay = 1/4000 # 1/128/15
        # Adam learning rate is *1/(1+n*decay) after n fits
    self.epsilon = 0 # 1
        # Exploration rate. If it is zero, it stays that way.
    self.epsilon_min = 0 # 0.01
        # Minimal value of epsilon.
    self.epsilon_decay = -1/2000 # -1/40
        # -! Exponential decay of epsilon towards epsilon_min
    self.use_two_epsilon = False
        # Use different epsilon when bomb is near!
    self.epsilon_bomb = 0 # 1
        # Exploration rate if bomb is close. If it is zero, it stays that way.
    self.epsilon_bomb_min = 0 # 0.01
        # Minimal value of epsilon_bomb.
    self.epsilon_bomb_decay = -1/800 # -1/40
        # -! Exponential decay of epsilon_bomb towards epsilon_bomb_min
    self.adjust_epsilon = False
        # False if epsilon and epsilon_bomb should be constant, if epsilon or epsilon_bomb is zero it stays that way.
    self.replay_batch_size = 32 # 32
        # Number of events to replay after each round.
    self.replay_batch_times = 0 # 128
        # Number of times to replay after each round.
    self.update_every = 200 # 256
        # The weights of the target model are updated every ... fits.
    self.train_every = 4
        # Replay after every ... steps
    self.start_training = 0
        # Start training after ... rounds, use the pervious ones to fill up the memory
    
    
    
    
    ####### Stuff that can stay this way
    
    # Load experience
    self.random_coin = False
        # Load experience_random_coin.npz
    self.random_crate = False
        # Load experience_random_crate.npz
    self.simple_crate = False
        # Load experience_simple_crate.npz
    
    # Actions
    self.actions = np.asarray(['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB'])
        # Better s[13] (getting the actions from settings), but the order works better this way.
        # Easier to exclude BOMB while using WAIT, also the order of LEFT/RIGHT and UP/DOWN here is the same as in self.events.
    self.actions_dict = dict([(self.actions[i],i) for i in range(self.actions.size)])
        # Dictionary to get a numerical value for "UP" etc.
    self.actions_rotate90_dict = {0:2, 1:3, 2:1, 3:0, 4:4, 5:5}
        # Dictionary for rotating actions clockwise by 90°. LEFT -> UP etc.
    self.shift_dict={2:np.reshape([0,0,-1,0],(4,1)),3:np.reshape([0,0,1,0],(4,1)),0:np.reshape([0,0,0,-1],(4,1)),
                        1:np.reshape([0,0,0,1],(4,1)),4:np.reshape([0,0,0,0],(4,1)),5:np.reshape([0,0,0,0],(4,1))}
        # Dictionary for actions
    
    # Collect values for the reward function
    self.rewards = np.asarray([self.value_walk]*4+[self.value_wait]+[0]+[self.value_impossible]\
                        +[self.value_bomb_drop]+[self.value_bomb_explode]+[self.value_crate]\
                        +[self.value_coin_found]+[self.value_coin]+[self.value_killed_opp]\
                        +[self.value_killed_self]+[self.value_got_killed]+[self.value_opp_elim]\
                        +[self.value_survived]+[self.value_steps]) # Collect rewards in array
    
    # Memory to evaluate performance, count rounds etc.
    self.edict = dict(zip(e._asdict().values(), e._asdict().keys()))
        # Dictionary for events
    self.aux_reward_name = np.asarray(["USELESS WAIT","USELESS BOMB","WAIT IN RANGE BOMB","STUCK BOMB","LOOP"])
        # Names of aux rewards
    self.chosenactions = np.full((self.action_size),0)
        # How often a specific action was chosen in this round
    self.count_events_size = 23
    self.count_events = np.full((self.count_events_size),0)
        # Contains all events of this round, sum of rewards, point allocated by game environment, forcecount
    self.counter = 0
        # Counter for rounds to save intermediate models
    self.fitcount = 0
        # Counter for fits to calculate learning rate
    self.forcecount = 0
        # Counter for forced
    self.learning_rate_adjusted = 0
        # Adjusted learning rate
    self.useless_bombs = 0
        # Count useless bombs
    self.reward_flow = deque()
        # Save count_events
    self.choice_flow = deque()
        # Save chosenactions
    self.parameters_flow = deque()
        # Save parameters
    self.step_counter = 1
        # Counter for overall steps
    
    # Memory for events
    self.memory = {"OLD_STATE":np.full((1,)+self.state_size,0), "ACTION":4,
                    "NEW_STATE":np.full((1,)+self.state_size,1), "REWARD":0,
                    "OLD_OLD_STATE":np.full((1,)+self.state_size,-1),
                    "USELESS_WAIT":False, "USELESS_OSC":False, "LOOP_DETECTED":False,
                    "OLD_RANDOM":True, "OLD_OLD_RANDOM": True, "NEW_RANDOM":True,
                    "RANDOM":0, "LAST_ACTION_RANDOM":False, "USELESS_OSC_REWARD": False,
                    "USELESS_WAIT_REWARD":False,"LAST_ACTION":4,
                    "LOOP":False}
    self.last_events = deque(maxlen=self.max_loop)
    self.last_events_random = deque(maxlen=self.max_loop)
        # Saving the last steps
    self.experience = Memory(self.experience_size, self)
        # SumTree with all steps
    self.training_data = deque(maxlen=self.training_data_size)
        # Deque with steps, either to generate random data or to train without PER.
    
    # Model
    # self.model = build_model(self, version=self.model_version)
        # # Build new model.
    # self.target_model = build_model(self, version=self.model_version)
        # # Build new target model.
    # update_target_model(self)
        # # Update the weights of the target model.
        # # Is this useful? could possibly give better results if not done.
    
    # MISC
    
    self.bomb_power = s.bomb_power
        # Strengh of the bombs
    
    # Load old memory
    if self.load_memory:
        load_sumtree(self)
    
    # If not learning, set epsilon=epsilon_min and epsilon_bomb=epsilon_min_bomb
    if not self.learning_bool: self.epsilon, self.epsilon_bomb = self.epsilon_min, self.epsilon_bomb_min
    
    # If generating random data, epsilon should stay 1.
    if self.generate_random_data:
        self.epsilon = 1
        self.epsilon_bomb = 1
        self.adjust_epsilon = False
    
    # # Load trained model
    # if not self.learning_bool or self.learning_bool_again:
        # # Load trained model, pass dictionary with custom layers!
        # # self.model = load_model("model/"+self.load_model+".h5", custom_objects={'DenseRandom': DenseRandom})
        # self.model.load_weights("model/"+self.load_weights+".h5")
        
        # # Loading the target model is only necessary when training should continue
        # # self.target_model = load_model("model/"+self.load_model+".h5", custom_objects={'DenseRandom': DenseRandom})
        # self.target_model.load_weights("model/"+self.load_weights+".h5")
    
    # Create folder
    if not os.path.exists(""+self.save_perf):
        os.makedirs(""+self.save_perf)
    
    # Prepopulate memory from saved experience, but only if learning
    if not self.generate_random_data and self.learning_bool:
        # Log
        self.logger.info('Populating the memory with loaded data')
        
        if self.random_crate:
            # Random starting data
            self.prepop_data_random_crate = np.load("experience_random_crate.npz")["random"]
            # Prepopulate experience with random data
            populate_experience(self, self.prepop_data_random_crate, self.usePER)
            # Huge waste of memory to still store the data!
            self.prepop_data_random_crate = None
        
        if self.random_coin:
            # Random starting data
            self.prepop_data_random_coin = np.load("experience_random_coin.npz")["random"]
            # Prepopulate experience with random data
            populate_experience(self, self.prepop_data_random_coin, self.usePER)
            # Huge waste of memory to still store the data!
            self.prepop_data_random_coin = None
        
        if self.simple_crate:
            # Starting data from simple_agent
            self.prepop_data_simple_crate = np.load("experience_simple_crate.npz")["simple"]
            # Prepopulate experience with data generated by simple agent
            populate_experience(self, self.prepop_data_simple_crate, self.usePER)
            # Huge waste of memory to still store the data!
            self.prepop_data_simple_crate = None
        
        # Log
        self.logger.info('Done')
    
    

def act(self):
    # Create state
    state = create_state(self, version=self.state_version)
    
    # First step?
    if self.game_state['step'] == 1:
        # Initialize memory at the beginning of every round
        first_round_reset(self,state)
        
        # Compute learning rate. Adam automatically adjusts it, this is just for saving it
        self.learning_rate_adjusted = self.learning_rate/(1+self.learning_rate_decay*self.fitcount)
        
        # Save the configuration once at the beginning
        if self.counter == 0 and not self.generate_random_data:
            save_config(self)
        
        # Count round
        self.counter += 1
        
        # Print round counter
        print("Round:      "+str(self.counter))
    
    # Choose action
    choose_simple(self)
    
    #self.step_counter += 1
    
    return # Important! If act does not return, main takes the maximal thinking time as the time it took us to decide!?


def reward_update(self):
    # Remember step, either to SumTree or deque
    #tree_bool, deque_bool = get_bools_remember(usePER=self.usePER, random=self.generate_random_data)
    remember(self, tree=False, deque=False, state_version=self.state_version)
    
    # Do not train if not training
    if self.generate_random_data or not self.learning_bool or self.counter < self.start_training: return
    
    # Replay one batch every n rounds
    if self.step_counter % self.train_every == 0:
        replay(self, batch_size=self.replay_batch_size, update=False, PER=self.usePER)
        

def end_of_episode(self):
    # Remember step, either to SumTree or deque
    #tree_bool, deque_bool = get_bools_remember(usePER=self.usePER, random=self.generate_random_data)
    # Only count as terminal state if everything was done!
    done = self.game_state['step'] != s[15]
    remember(self, tree=False, deque=False, state_version=self.state_version, done=done)
    
    # # Generate random data?
    # if self.generate_random_data:
        # print("                 "+str(len(self.training_data)))
        # # If last round
        # if self.counter == s[6]:
            # # Save compressed to save space
            # np.savez_compressed(""+self.store_random_as+".npz",random=self.training_data)
        # return
    
    # Print stuff
    print_reset_stuff(self)
    
    # Save evaluation parameters every 10 rounds
    if self.counter % 10 == 0:
        np.save(""+self.save_perf+"/reward_flow.npy",self.reward_flow)
        np.save(""+self.save_perf+"/choice_flow.npy",self.choice_flow)
        np.save(""+self.save_perf+"/parameters_flow.npy",self.parameters_flow)
        
    # # Do not train if not training
    # if not self.learning_bool or self.counter < self.start_training: return
    
    # # Replay n batches after the round is finished
    # for i in range(self.replay_batch_times):
        # replay(self, batch_size=self.replay_batch_size, update=False, PER=self.usePER)
    
    # # Update the target model every n fits
    # if self.fitcount % self.update_every == 0 and self.use_target:
        # update_target_model(self)
    
    # # Save current weights
    # #self.model.save("model/"+self.save_as+".h5")
    # self.model.save_weights("model/"+self.save_as+"_weights.h5")
    
    # # Reduce epsilon
    # adjust_epsilon(self)
        
    # # Save the model weights every 20 rounds to track evolution
    # if self.counter % 20 == 0:
        # #self.model.save("model/"+self.save_as+"N"+str(self.counter)+".h5")
        # self.model.save_weights("model/"+self.save_as+"_weightsN"+str(self.counter)+".h5")
        
    # # Save the memory every 250 rounds to allow restarting the learning?
    # if self.save_memory and self.counter % 250 == 0:
        save_sumtree(self)


def create_state(self, version="ALLOHL"):
    """ Create state from self.game_state[''].
        
        Only "SMALL", "ALL" and "ALLOH" are useful!
        
        "SMALL":
            Returns array of size (1,3,17,17).
            Contains arrays for the arena, the coins and the player position.
        "ALL":
            Returns array of size (1,6,17,17).
            The same as "SMALL", but adds arrays for bombs and other players.
        "ALLOH":
            Returns array of size (1,9,17,17).
            The same as "ALL", but (alomst) everything is encoded as One-hot.
        "ALLOHL":
            Returns array of size (1,13,17,17).
            The same as "ALLOH", but bombs and explosions encoded as One-hot.
            
        "ALLCOND":
            Returns array of size (1,1,2*17,3*17).
            The information from "ALL" is condensed into an array of (2,3) fields with the information
            of the respective square of the playing field.
        "SMALLCOLL":
            Returns array of size (1,3*176).
            For a normal Dense layer to work, the input has to be onedimensional.
            Collapsed the information from SMALL into one dimension, again with added dimension for batch.
            Irrelevant d.o.f (arena==-1) are deleted, since they are not varying.
        "ALLCOLL"
            Returns array of size (1,6*176).
            The same as "SMALLCOLL", but for "ALL".
        "AUGMENTED":
            Not ready yet, should add more parameters, like index indicating
            in which direction we can move etc. I am not sure this would be helpful?
        
        The extra dimensions are necessary because the convolutional nets expect
        the input to be of shape (batch, channels, rows, cols).
        
        The if queries are necesary because array[()]=array, so an empty list of
        coin positions etc. otherwise gives wrong results.
        
        state_size can never be just (n), take (n,) since this is iterable.
    """
    
    if version == "SMALL":
        # Arena
        arena = np.asarray(self.game_state['arena'])
        # Self
        position = np.asarray(self.game_state['self'])[:2].astype(int) # For now only the position
        positions = np.full((17,17), 0) # Empty Arena
        positions[tuple(position)] = 1 # 1 for self
        # Coins
        coins = np.full((17,17), 0) # Empty Arena
        coinindex = self.game_state['coins']
        if len(coinindex)>0: coins[tuple(zip(*coinindex))] = 1 # =1 for coin
        # Collect as state
        state = np.expand_dims(np.stack([arena,positions,coins],0),0)
        return state
    
    elif version == "ALL":
        # Arena
        arena = np.asarray(self.game_state['arena'])
        # Players
        choose = [0,1,3]
        positionself = np.asarray(self.game_state['self'])[choose].astype(int)
        positionothers = []
        if len(self.game_state['others'])>0: positionothers = np.asarray(self.game_state['others'])[:,choose].astype(int)
        positions = np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: positions[tuple(positionself[:2])] = 1 # 1 for self
        if len(positionothers)>0: positions[tuple(zip(*positionothers[:,:2]))] = -1 # -1 for others
        bombpossible = np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: bombpossible[tuple(positionself[:2])] = positionself[2] # 1 for possible, 0 for not possible
        if len(positionothers)>0: bombpossible[tuple(zip(*positionothers[:,:2]))] = positionothers[:,2]
        # Bombs
        bombs = np.full((17,17), 0) # Empty Arena
        positionbombs = np.asarray(self.game_state['bombs']).astype(int)
        # +1 is crucial! Otherwise, the step before the bomb explodes does not contain a noticeable bomb!
        if len(positionbombs)>0: bombs[tuple(zip(*positionbombs[:,:2]))] = positionbombs[:,2] + 1
        # Explosions
        explosions = np.asarray(self.game_state['explosions'])
        # Coins
        coins = np.full((17,17), 0) # Empty Arena
        coinindex = self.game_state['coins']
        if len(coinindex)>0: coins[tuple(zip(*coinindex))] = 1 # =1 for coin
        # Collect as state
        state = np.expand_dims(np.stack([arena,positions,bombpossible,bombs,explosions,coins],0),0)
        return state
    
    elif version == "ALLOH":
        # Arena
        arena = np.asarray(self.game_state['arena'])
        wall, crate = np.zeros((17,17)), np.zeros((17,17))
        wall[arena==-1], crate[arena==1] = 1, 1
        # Players
        choose = [0,1,3]
        positionself = np.asarray(self.game_state['self'])[choose].astype(int)
        positionothers = []
        if len(self.game_state['others'])>0: positionothers = np.asarray(self.game_state['others'])[:,choose].astype(int)
        positionsself, positionsother = np.full((17,17), 0), np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: positionsself[tuple(positionself[:2])] = 1 # 1 for self
        if len(positionothers)>0: positionsother[tuple(zip(*positionothers[:,:2]))] = 1 # 1 for others
        bombpossibleself, bombpossibleother = np.full((17,17), 0), np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: bombpossibleself[tuple(positionself[:2])] = positionself[2] # 1 for possible, 0 for not possible
        if len(positionothers)>0: bombpossibleother[tuple(zip(*positionothers[:,:2]))] = positionothers[:,2]
        # Bombs
        bombs = np.full((17,17), 0) # Empty Arena
        positionbombs = np.asarray(self.game_state['bombs']).astype(int)
        # +1 is crucial! Otherwise, the step before the bomb explodes does not contain a noticeable bomb!
        if len(positionbombs)>0: bombs[tuple(zip(*positionbombs[:,:2]))] = positionbombs[:,2] + 1
        # Explosions
        explosions = np.asarray(self.game_state['explosions'])
        # Coins
        coins = np.full((17,17), 0) # Empty Arena
        coinindex = self.game_state['coins']
        if len(coinindex)>0: coins[tuple(zip(*coinindex))] = 1 # =1 for coin
        # Collect as state
        state = np.expand_dims(np.stack([wall,crate,coins,positionsself,bombpossibleself,bombs,explosions,positionsother,bombpossibleother],0),0)
        return state
    
    elif version == "ALLOHL":
        # Arena
        arena = np.asarray(self.game_state['arena'])
        wall, crate = np.zeros((17,17)), np.zeros((17,17))
        wall[arena==-1], crate[arena==1] = 1, 1
        # Players
        choose = [0,1,3]
        positionself = np.asarray(self.game_state['self'])[choose].astype(int)
        positionothers = []
        if len(self.game_state['others'])>0: positionothers = np.asarray(self.game_state['others'])[:,choose].astype(int)
        positionsself, positionsother = np.full((17,17), 0), np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: positionsself[tuple(positionself[:2])] = 1 # 1 for self
        if len(positionothers)>0: positionsother[tuple(zip(*positionothers[:,:2]))] = 1 # 1 for others
        bombpossibleself, bombpossibleother = np.full((17,17), 0), np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: bombpossibleself[tuple(positionself[:2])] = positionself[2] # 1 for possible, 0 for not possible
        if len(positionothers)>0: bombpossibleother[tuple(zip(*positionothers[:,:2]))] = positionothers[:,2]
        # Bombs
        bombs = np.full((17,17), 0) # Empty Arena
        positionbombs = np.asarray(self.game_state['bombs']).astype(int)
        # +1 is crucial! Otherwise, the step before the bomb explodes does not contain a noticeable bomb!
        if len(positionbombs)>0: bombs[tuple(zip(*positionbombs[:,:2]))] = positionbombs[:,2] + 1
        bombsOH = np.full((4,17,17), 0) # Empty Arenas
        bombsOH[0][bombs==1], bombsOH[1][bombs==2], bombsOH[2][bombs==3], bombsOH[3][bombs==4] = [1]*4
        # Explosions
        explosions = np.asarray(self.game_state['explosions'])
        explosionsOH = np.full((2,17,17), 0) # Empty Arenas
        explosionsOH[0][explosions==1], explosionsOH[1][explosions==2] = [1]*2
        # Coins
        coins = np.full((17,17), 0) # Empty Arena
        coinindex = self.game_state['coins']
        if len(coinindex)>0: coins[tuple(zip(*coinindex))] = 1 # =1 for coin
        # Collect as state
        state = np.expand_dims(np.stack([wall,crate,coins,positionsself,bombpossibleself,*bombsOH,*explosionsOH,positionsother,bombpossibleother],0),0)
        return state
    
    elif version == "ALLSPARSE":
        # Arena
        arena = np.asarray(self.game_state['arena'])
        # Players
        choose = [0,1,3]
        positionself = np.asarray(self.game_state['self'])[choose].astype(int)
        positionothers = []
        if len(self.game_state['others'])>0: positionothers = np.asarray(self.game_state['others'])[:,choose].astype(int)
        positions = np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: positions[tuple(positionself[:2])] = 1 # 1 for self
        if len(positionothers)>0: positions[tuple(zip(*positionothers[:,:2]))] = -1 # -1 for others
        bombpossible = np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: bombpossible[tuple(positionself[:2])] = positionself[2] # 1 for possible, 0 for not possible
        if len(positionothers)>0: bombpossible[tuple(zip(*positionothers[:,:2]))] = positionothers[:,2]
        # Bombs
        bombs = np.full((17,17), 0) # Empty Arena
        positionbombs = np.asarray(self.game_state['bombs']).astype(int)
        # +1 is crucial! Otherwise, the step before the bomb explodes does not contain a noticeable bomb!
        if len(positionbombs)>0: bombs[tuple(zip(*positionbombs[:,:2]))] = positionbombs[:,2] + 1
        # Explosions
        explosions = np.asarray(self.game_state['explosions'])
        # Coins
        coins = np.full((17,17), 0) # Empty Arena
        coinindex = self.game_state['coins']
        if len(coinindex)>0: coins[tuple(zip(*coinindex))] = 1 # =1 for coin
        # Collect as state
        state = np.expand_dims(np.stack([arena,positions,bombpossible,bombs,explosions,coins],0),0)
        return sparse.COO(np.where(state!=0),state[state!=0],state.shape)
    
    elif version == "ALLCOND":
        # Arena
        arena = np.asarray(self.game_state['arena'])
        # Players
        choose = [0,1,3]
        positionself = np.asarray(self.game_state['self'])[choose].astype(int)
        positionothers = []
        if len(self.game_state['others'])>0: positionothers = np.asarray(self.game_state['others'])[:,choose].astype(int)
        positions = np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: positions[tuple(positionself[:2])] = 1 # 1 for self
        if len(positionothers)>0: positions[tuple(zip(*positionothers[:,:2]))] = -1 # -1 for others
        bombpossible = np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: bombpossible[tuple(positionself[:2])] = positionself[2] # 1 for possible, 0 for not possible
        if len(positionothers)>0: bombpossible[tuple(zip(*positionothers[:,:2]))] = positionothers[:,2]
        # Bombs
        bombs = np.full((17,17), 0) # Empty Arena
        positionbombs = np.asarray(self.game_state['bombs']).astype(int)
        # +1 is crucial! Otherwise, the step before the bomb explodes does not contain a noticeable bomb!
        if len(positionbombs)>0: bombs[tuple(zip(*positionbombs[:,:2]))] = positionbombs[:,2] + 1
        # Explosions
        explosions = np.asarray(self.game_state['explosions'])
        # Coins
        coins = np.full((17,17), 0) # Empty Arena
        coinindex = self.game_state['coins']
        if len(coinindex)>0: coins[tuple(zip(*coinindex))] = 1 # =1 for coin
        # Collect as state
        state = np.full((2*17,3*17),0)
        state[0::2,0::3] = arena
        state[0::2,1::3] = positions
        state[0::2,2::3] = bombpossible
        state[1::2,0::3] = coins
        state[1::2,1::3] = explosions
        state[1::2,2::3] = bombs
        state = np.expand_dims(state,0)
        return state
    
    elif version == "SMALLCOLL":
        # Arena
        arena = np.asarray(self.game_state['arena'])
        # Self
        position = np.asarray(self.game_state['self'])[:2].astype(int) # For now only the position
        positions = np.full((17,17), 0) # Empty Arena
        positions[tuple(position)] = 1 # 1 for self
        # Coins
        coins = np.full((17,17), 0) # Empty Arena
        coinindex = self.game_state['coins']
        if len(coinindex)>0: coins[tuple(zip(*coinindex))] = 1 # =1 for coin
        # Collect as flattened state
        state = np.expand_dims(np.hstack([arena[arena!=-1],positions[arena!=-1],coins[arena!=-1]]),0)
        return state
    
    elif version == "ALLCOLL":
        # Arena
        arena = np.asarray(self.game_state['arena'])
        # Players
        choose = [0,1,3]
        positionself = np.asarray(self.game_state['self'])[choose].astype(int)
        positionothers = []
        if len(self.game_state['others'])>0: positionothers = np.asarray(self.game_state['others'])[:,choose].astype(int)
        positions = np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: positions[tuple(positionself[:2])] = 1 # 1 for self
        if len(positionothers)>0: positions[tuple(zip(*positionothers[:,:2]))] = -1 # -1 for others
        bombpossible = np.full((17,17), 0) # Empty Arena
        if len(positionself)>0: bombpossible[tuple(positionself[:2])] = positionself[2] # 1 for possible, 0 for not possible
        if len(positionothers)>0: bombpossible[tuple(zip(*positionothers[:,:2]))] = positionothers[:,2]
        # Bombs
        bombs = np.full((17,17), 0) # Empty Arena
        positionbombs = np.asarray(self.game_state['bombs']).astype(int)
        # +1 is crucial! Otherwise, the step before the bomb explodes does not contain a noticeable bomb!
        if len(positionbombs)>0: bombs[tuple(zip(*positionbombs[:,:2]))] = positionbombs[:,2] + 1
        # Explosions
        explosions = np.asarray(self.game_state['explosions'])
        # Coins
        coins = np.full((17,17), 0) # Empty Arena
        coinindex = self.game_state['coins']
        if len(coinindex)>0: coins[tuple(zip(*coinindex))] = 1 # =1 for coin
        # Collect as state
        state = np.expand_dims(np.hstack([arena[arena!=-1],positions[arena!=-1],bombpossible[arena!=-1],bombs[arena!=-1],explosions[arena!=-1],coins[arena!=-1]]),0)
        return state
    
    elif version == "AUGMENTED":
        # Arena
        arena = np.asarray(self.game_state['arena'])
        # Self
        position = np.asarray(self.game_state['self'])[:2].astype(int) # For now only the position
        positions = np.full((17,17), 0) # Empty Arena
        positions[tuple(position)] = 1 # 1 for self, later -1 for others
        # Walk possible? Extra parameters, maybe makes it easier for the NN.
        # positionborder = position + np.asarray([[0,1],[1,0],[0,-1],[-1,0]])
        # walkpossible = np.asarray(arenagrid[tuple(zip(*positionborder))])
        # walkpossible[walkpossible!=0] = 1 # =1 if impossible to walk in that direction.
        # Coins
        coins = np.full((17,17), 0) # Empty Arena
        coins[tuple(zip(*self.game_state['coins']))] = 1 # =1 for coin
        # Collect as state
        state = np.expand_dims(np.stack([arena,positions,coins],0),0) # The same as np.asarray([state]) later. Extra dim necessary because model expects batches of states
        return state
    
    else:
        raise ValueError("There is no predefined state with this name!", version)


def get_reward(self, count=True):
    """ Returns the reward for the last step based on self.events.
        For the counting of occured events to work, this should be called
        with count=True exactly once after every round.
        Relies on self.rewards for the rewards of specific actions.
    """
    # Get events
    events = np.asarray(self.events) # asarray necessary for == to work properly
    # Get occurences of events
    counts = np.append(np.count_nonzero([events==i for i in range(17)],axis=1),[self.game_state['step']]) # Normally range(e.size). Returns counts for every action
    # Compute reward
    reward = np.dot(counts,self.rewards)
    # Count events
    if count: self.count_events[:18] += np.append(counts[:17],[reward]) # Normally [:e.size]
    # Return reward
    return reward, counts

    
def choose_action(self,state):
    """ Set next action."""
    # Force the agent out of oscillations
    if self.force_random:
        # It should still take random actions
        if self.memory["RANDOM"] != 0:
            # Save who chose this action
            self.memory["LAST_ACTION_RANDOM"] = True
            # Count forced actions
            self.forcecount += 1
            # Reduce the counter for forced actions
            self.memory["RANDOM"] -= 1
            # Get possible actions
            possible = actions_possible(self,state,self.explosions_impossible_random)
            # Copy pactions, otherwise we change the values permanently!
            prob = np.copy(self.pactions)
            # Set the probability for impossible actions to zero
            prob[np.invert(possible[0])]=0
            # Standardize the probabilities, np.random.choice needs this
            # Not necessarily exactly one, but np.random.choice hat a built in tolerance
            prob=prob/np.sum(prob)
            # Choose an actions according to the probabilities
            self.next_action = np.random.choice(self.actions, p=prob)
            # Log
            self.logger.info('RANDOM FORCED')
            self.logger.debug(self.next_action)
        
        # Useless waiting
        elif self.memory["USELESS_WAIT"]:
            # Save who chose this action
            self.memory["LAST_ACTION_RANDOM"] = True
            # Count forced actions
            self.forcecount += 1
            # Force the next actions to be random
            self.memory["RANDOM"] = self.random_actions_after_wait -1
            # Get possible actions
            possible = actions_possible(self,state,self.explosions_impossible_random)
            # Do not wait if waiting is useless
            possible[0,4] = False
            # Copy pactions, otherwise we change the values permanently!
            prob = np.copy(self.pactions)
            # Set the probability for impossible actions to zero
            prob[np.invert(possible[0])]=0
            # Standardize the probabilities, np.random.choice needs this
            # Not necessarily exactly one, but np.random.choice hat a built in tolerance
            prob=prob/np.sum(prob)
            # Choose an actions according to the probabilities
            self.next_action = np.random.choice(self.actions, p=prob)
            # Log
            self.logger.info('RANDOM FORCED')
            self.logger.debug(self.next_action)
        
        # Stuck in oscillation, maybe do not allow all actions here?
        elif self.memory["USELESS_OSC"]:
            # Save who chose this action
            self.memory["LAST_ACTION_RANDOM"] = True
            # Count forced actions
            self.forcecount += 1
            # Force the next actions to be random
            self.memory["RANDOM"] = self.random_actions_after_osc -1
            # Get possible actions
            possible = actions_possible(self,state,self.explosions_impossible_random)
            # Bad idea, gives poor results!
                # Do not go back into oscillation!
                # Should work with and without crates?
                #possible[0,self.actions_rotate90_dict[self.actions_rotate90_dict[self.memory["ACTION"]]]] = False
            # Copy pactions, otherwise we change the values permanently!
            prob = np.copy(self.pactions)
            # Set the probability for impossible actions to zero
            prob[np.invert(possible[0])]=0
            # Standardize the probabilities, np.random.choice needs this
            # Not necessarily exactly one, but np.random.choice hat a built in tolerance
            prob=prob/np.sum(prob)
            # Choose an actions according to the probabilities
            self.next_action = np.random.choice(self.actions, p=prob)
            # Log
            self.logger.info('RANDOM FORCED')
            self.logger.debug(self.next_action)
        
        # Loop
        elif self.memory["LOOP"]:
            # Save who chose this action
            self.memory["LAST_ACTION_RANDOM"] = True
            # Reset LOOP
            self.memory["LOOP"] = False
            # Count forced actions
            self.forcecount += 1
            # Force the next actions to be random
            self.memory["RANDOM"] = self.random_after_loop -1
            # Get possible actions
            possible = actions_possible(self,state,self.explosions_impossible_random)
            # Copy pactions, otherwise we change the values permanently!
            prob = np.copy(self.pactions)
            # Set the probability for impossible actions to zero
            prob[np.invert(possible[0])]=0
            # Standardize the probabilities, np.random.choice needs this
            # Not necessarily exactly one, but np.random.choice hat a built in tolerance
            prob=prob/np.sum(prob)
            # Choose an actions according to the probabilities
            self.next_action = np.random.choice(self.actions, p=prob)
            # Log
            self.logger.info('RANDOM FORCED LOOP')
            self.logger.debug(self.next_action)
        
        # Otherwise use epsilon-greedy
        else:
            epsilon = self.epsilon
            # If two epsilons should be used and bomb is near, use epsilon_bomb
            if self.use_two_epsilon and is_bomb_near(self,state):
                epsilon = self.epsilon_bomb
                # Log
                self.logger.info('EPSILONBOMB')
            
            # Either choose randomly
            if np.random.rand() <= epsilon:
                # Save who chose this action
                self.memory["LAST_ACTION_RANDOM"] = True
                # Get possible actions
                possible = actions_possible(self,state,self.explosions_impossible_random)
                # Not Bomb as first action
                if self.first_not_bomb and self.game_state['step'] == 1:
                    possible[0,5] = False
                # Copy pactions, otherwise we change the values permanently!
                prob = np.copy(self.pactions)
                # Set the probability for impossible actions to zero
                prob[np.invert(possible[0])]=0
                # Standardize the probabilities, np.random.choice needs this
                # Not necessarily exactly one, but np.random.choice hat a built in tolerance
                prob=prob/np.sum(prob)
                # Choose an actions according to the probabilities
                self.next_action = np.random.choice(self.actions, p=prob)
                # Log
                self.logger.info('RANDOM')
                self.logger.debug(self.next_action)
            
            # Or according to our model
            else:
                # Save who chose this action
                self.memory["LAST_ACTION_RANDOM"] = False
                # Not Bomb as first action
                if self.first_not_bomb and self.game_state['step'] == 1:
                    # Choose the possible action with the highest expected reward.
                    next, values, possible = predict_action(self,state,first=True)
                else:
                    # Choose the possible action with the highest expected reward.
                    next, values, possible = predict_action(self,state)
                self.next_action = self.actions[next[0]]
                # Log
                self.logger.info('MODEL')
                self.logger.debug(values[0])
                self.logger.debug(self.next_action)
                # Store the action only if it was chosen by the model.
                self.chosenactions[next[0]] += 1
    # Or not
    else:
        epsilon = self.epsilon
        # If two epsilons should be used and bomb is near, use epsilon_bomb
        if self.use_two_epsilon and is_bomb_near(self,state):
            epsilon = self.epsilon_bomb
            # Log
            self.logger.info('EPSILONBOMB')
        
        # Either choose randomly
        if np.random.rand() <= epsilon:
            # Save who chose this action
            self.memory["LAST_ACTION_RANDOM"] = True
            # Get possible actions
            possible = actions_possible(self,state,self.explosions_impossible_random)
            # Not Bomb as first action
            if self.first_not_bomb and self.game_state['step'] == 1:
                possible[0,5] = False
            # Copy pactions, otherwise we change the values permanently!
            prob = np.copy(self.pactions)
            # Set the probability for impossible actions to zero
            prob[np.invert(possible[0])]=0
            # Standardize the probabilities, np.random.choice needs this
            # Not necessarily exactly one, but np.random.choice hat a built in tolerance
            prob=prob/np.sum(prob)
            # Choose an actions according to the probabilities
            self.next_action = np.random.choice(self.actions, p=prob)
            # Log
            self.logger.info('RANDOM')
            self.logger.debug(self.next_action)
        
        # Or according to our model
        else:
            # Save who chose this action
            self.memory["LAST_ACTION_RANDOM"] = False
            # Not Bomb as first action
            if self.first_not_bomb and self.game_state['step'] == 1:
                # Choose the possible action with the highest expected reward.
                next, values, possible = predict_action(self,state,first=True)
            else:
                # Choose the possible action with the highest expected reward.
                next, values, possible = predict_action(self,state)
            self.next_action = self.actions[next[0]]
            # Log
            self.logger.info('MODEL')
            self.logger.debug(values[0])
            self.logger.debug(self.next_action)
            # Store the action only if it was chosen by the model.
            self.chosenactions[next[0]] += 1


def remember(self, done=False, tree=True, deque=False, state_version="ALLOHL"):
    """ Remember actions, and get whether they are useless.
    """
    # Log events
    self.logger.info([self.edict[ev] for ev in self.events])
    # Shift NEW and OLD one step back
    self.memory["OLD_OLD_STATE"] = self.memory["OLD_STATE"]
    self.memory["OLD_OLD_RANDOM"] = self.memory["OLD_RANDOM"]
    self.memory["OLD_STATE"] = self.memory["NEW_STATE"]
    self.memory["OLD_RANDOM"] = self.memory["NEW_RANDOM"]
    # Save this step
    self.memory["NEW_STATE"] = create_state(self, state_version)
    self.memory["NEW_RANDOM"] = self.memory["LAST_ACTION_RANDOM"]
    # Save to loop deque
    self.last_events.appendleft(self.memory["NEW_STATE"])
    self.last_events_random.appendleft(self.memory["LAST_ACTION_RANDOM"])
    # Did the agent (not random) wait, and the state did not change (useless waiting)?
    self.memory["USELESS_WAIT_REWARD"] = states_equal(self,self.memory["NEW_STATE"],self.memory["OLD_STATE"])
    self.memory["USELESS_WAIT"] = self.memory["USELESS_WAIT_REWARD"] and not self.memory["NEW_RANDOM"]
    # Did the agent (not random) choose two consecutive actions that returned it to the same state as before (useless oscillation)?
    self.memory["USELESS_OSC_REWARD"] = states_equal(self,self.memory["NEW_STATE"],self.memory["OLD_OLD_STATE"])
    self.memory["USELESS_OSC"] = self.memory["USELESS_OSC_REWARD"] and not self.memory["OLD_RANDOM"] and not self.memory["NEW_RANDOM"]
    # Search for loops
    if len(self.last_events_random) >= self.max_loop and self.prevent_loops and not self.memory["LAST_ACTION_RANDOM"]:
        actions_random = [np.all([not self.last_events_random[i] for i in range(self.min_loop+j)]) for j in range(self.max_loop-self.min_loop+1)]
        loops = [np.array_equal(self.last_events[0],self.last_events[self.min_loop+j-1]) for j in range(self.max_loop-self.min_loop+1)]
        self.memory["LOOP"] = np.count_nonzero(np.logical_and(actions_random,loops))>0
    # Save action
    self.memory["LAST_ACTION"] = self.memory["ACTION"]
    self.memory["ACTION"] = self.actions_dict[self.next_action]
    # Save Reward
    self.memory["REWARD"], events = get_reward(self, count=True)
    # Extra reward for useless waiting
    if self.value_useless_waiting != 0:
        if self.memory["USELESS_WAIT_REWARD"]: self.memory["REWARD"] += self.value_useless_waiting
    # Extra reward for oscillation
    if self.value_osc != 0:
        if self.memory["USELESS_OSC_REWARD"]: self.memory["REWARD"] += self.value_osc
    # Extra reward for loop
    if self.value_loop != 0:
        if self.memory["LOOP"]: self.memory["REWARD"] += self.value_loop
    # Extra reward for useless bomb
    # Useless means bomb exploded, but destroyed no crate and killed no opponent
    useless_bomb = False
    if self.value_useless_bomb != 0:
        useless_bomb = events[8] != 0 and events[9] == 0 and  events[12] == 0
        if useless_bomb:
            self.memory["REWARD"] += self.value_useless_bomb
            self.useless_bombs += 1
    # Extra reward for waiting while standing in range of a ticking bomb
    wait_range_bomb = False
    if self.value_wait_range_bomb != 0 and not self.no_waiting_bomb_range:
        wait_range_bomb = self.next_action == "WAIT" and in_range_of_bomb(self,self.memory["OLD_STATE"])
        if wait_range_bomb: self.memory["REWARD"] += self.value_wait_range_bomb
    # Extra reward for getting stuck in range of a bomb
    stuck_bomb = False
    if self.stuck_bomb != 0:
        possible = actions_possible(self,self.memory["NEW_STATE"],False)
        stuck_bomb = np.count_nonzero(possible) == 1 and possible[0,4] == True and in_range_of_bomb(self,self.memory["NEW_STATE"]) and self.memory["LAST_ACTION"] == 5
        if stuck_bomb: self.memory["REWARD"] += self.stuck_bomb
    # Log auxiliary rewards
    self.logger.debug(self.aux_reward_name[np.asarray([self.memory["USELESS_WAIT_REWARD"],useless_bomb,wait_range_bomb,stuck_bomb,self.memory["LOOP"]])])
    # Add this step to experience
    if tree: self.experience.store((self.memory["OLD_STATE"], self.memory["ACTION"], self.memory["REWARD"], self.memory["NEW_STATE"], done))
    # Add this step to training_data
    if deque: self.training_data.append((self.memory["OLD_STATE"], self.memory["ACTION"], self.memory["REWARD"], self.memory["NEW_STATE"], done))


def update_target_model(self):
    """ Update the target model weights with the model weights.
    """
    self.target_model.set_weights(self.model.get_weights())


def print_reset_stuff(self, print_bool=True):
    """ Print stuff to evaluate performance, and reset the respective counters.
    """
    # Print chosen actions
    if print_bool: print("Chosen: "+str(self.chosenactions)+"    "+str(self.actions[:self.action_size]))
    # Save chosen actions
    self.choice_flow.append(self.chosenactions)
    # Reset chosen actions
    self.chosenactions = np.full((self.action_size),0)
    # Add stuff to count_events
    self.count_events[18] = self.game_state['self'][4]
        # Points from the game
    self.count_events[19] = self.game_state['step']
        # Steps till termination
    self.count_events[20] = self.forcecount
        # How often random actions were forced
    self.count_events[21] = self.fitcount
        # How often replay was called
    self.count_events[22] = self.useless_bombs
        # 
    # Reset forcecount, useless_bombs
    self.forcecount = 0
    self.useless_bombs = 0
    # Print events
    if print_bool: print("Events: "+str(self.count_events[:17]))
    # Print rewards
    if print_bool: print("Reward:   "+str(self.count_events[17:20]))
    # Save events
    self.reward_flow.append(self.count_events)
    # Save variable parameters
    self.parameters_flow.append((self.epsilon,self.PER_b,self.learning_rate_adjusted,self.epsilon_bomb))
    # Reset events counter
    self.count_events = np.full((self.count_events_size),0)
    # Print number of steps till the round ended, epsilon, b, learning rate
    if print_bool: print("Steps: {}    e: {:.4f}    e_b: {:.4f}    PER_b: {:.4f}    LR: {:.8f}".format(self.game_state['step'],self.epsilon,self.epsilon_bomb,self.PER_b,self.learning_rate_adjusted))


def build_model(self, version = "DUELCONVSYM"):
    """ Generates the model.
        
        "DUELCONV":
            Convolutional layers with dueling networks for value and advantage.
        "DUELCONVSYM":
            The network of DUELCONV, but this version uses the symmetries of our state space. For every state,
            there are eight (four rotations, reflection of them) redundant versions that should result in the same
            action. Therefore, the Q values of a state are predicted as the average over all eight representations,
            and similarly backpropagation takes all of them in account.
        "DUELCONVSYMNOISE":
            Same as above, but all Dense layers are replaced by NoisyDense.
        "SIMPLE":
            Simple NN with Dense layers.
    """
    
    if version == "DUELCONV":
        # Input is input layer
        # Conv2D is 2D convolutional layer
        # Flatten is layer that collapses all dimensions into one
        # Dense is standard NN
        # Lambda applies function to inputs
        
        
        # Input is state
        input = Input(shape=self.state_size)
        # data_format="channels_first" so that input is (batch, channels, rows, cols) instead of (batch, rows, cols, channels)
        # Cond2D arguments: no of filters, size of the convolution window(here and next, (i,i) gives the same as (i)), strides over the picture
        # kernel 4x4, stride 1: 17x17 turns into 14x14
        shared = Conv2D(32, (4, 4), strides=(1), activation='relu', data_format="channels_first")(input)
        # kernel 4x4, stride 2: 14x14 turns into 6x6
        shared = Conv2D(64, (4, 4), strides=(2), activation='relu', data_format="channels_first")(shared)
        # kernel 2x2, stride 1: 6x6 turns into 5x5
        shared = Conv2D(64, (3, 3), strides=(1), activation='relu', data_format="channels_first")(shared)
        # Turn output from above into flat layer
        flatten = Flatten()(shared)
        
        # NN to predict the best action
        # Turn flattened above to 256
        advantage_fc = Dense(256, activation='relu')(flatten)
        # Map above to actions
        advantage = Dense(self.action_size, activation='linear')(advantage_fc)
        # We dont want the absolute advantage but the relative one, so we subtract the mean of all actions from every action
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), # keepdims=True so we get action_size dimension with the mean everywhere
                           output_shape=(self.action_size,))(advantage)
        
        # NN to predict the value of the state
        # Turn flattened above to 256
        value_fc = Dense(256, activation='relu')(flatten)
        # Map above to one value, the value of state
        value =  Dense(1)(value_fc)
        # Expand this value into action_size dimensions to sum it with advantage
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(value)
        
        # Merge value and advantage into single output
        #q_value = merge([value, advantage], mode='sum')
        q_value = add([value, advantage])
        # Creates a model with .fit(), .predict() etc. from the layers.
        model = Model(inputs=input, outputs=q_value)
        # Compile the model
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        # Print a summary of the model
        #model.summary()
        # Trainable params: 597,413

        return model
    
    elif version == "DUELCONVI":
        # Input is input layer
        # Conv2D is 2D convolutional layer
        # Flatten is layer that collapses all dimensions into one
        # Dense is standard NN
        # Lambda applies function to inputs
        
        
        # Input is state
        input = Input(shape=self.state_size)
        # data_format="channels_first" so that input is (batch, channels, rows, cols) instead of (batch, rows, cols, channels)
        # Cond2D arguments: no of filters, size of the convolution window(here and next, (i,i) gives the same as (i)), strides over the picture
        # kernel 4x4, stride 1: 17x17 turns into 14x14
        shared = Conv2D(128, (4, 4), strides=(1), activation='elu', data_format="channels_first")(input)
        # kernel 4x4, stride 2: 14x14 turns into 6x6
        shared = Conv2D(64, (4, 4), strides=(2), activation='elu', data_format="channels_first")(shared)
        # kernel 2x2, stride 1: 6x6 turns into 5x5
        shared = Conv2D(32, (3, 3), strides=(1), activation='elu', data_format="channels_first")(shared)
        # Turn output from above into flat layer
        flatten = Flatten()(shared)
        
        # NN to predict the best action
        # Turn flattened above to 256
        advantage_fc = Dense(256, activation='elu')(flatten)
        # Map above to actions
        advantage = Dense(self.action_size, activation='linear')(advantage_fc)
        # We dont want the absolute advantage but the relative one, so we subtract the mean of all actions from every action
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), # keepdims=True so we get action_size dimension with the mean everywhere
                           output_shape=(self.action_size,))(advantage)
        
        # NN to predict the value of the state
        # Turn flattened above to 256
        value_fc = Dense(256, activation='elu')(flatten)
        # Map above to one value, the value of state
        value =  Dense(1)(value_fc)
        # Expand this value into action_size dimensions to sum it with advantage
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(value)
        
        # Merge value and advantage into single output
        #q_value = merge([value, advantage], mode='sum')
        q_value = add([value, advantage])
        # Creates a model with .fit(), .predict() etc. from the layers.
        model = Model(inputs=input, outputs=q_value)
        # Compile the model
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        # Print a summary of the model
        #model.summary()
        # Trainable params: 597,413

        return model
    
    elif version == "DUELCONVSYM":
        # Here, I assume a state of shape (?,m,N,N) where the last two indices represent the board!
        
        # Define a matrix that will return the mirror image of the matrix
        MirrorMatrix = K.variable(np.rot90(np.identity(self.state_size[-1]),1))
        # Define a function to mirror a keras tensor
        def mirror(x):
            return K.dot(x,MirrorMatrix)
        # Define a function to transpose a keras tensor.
        def transpose(x):
            return K.permute_dimensions(x,(0,1,3,2))
        # Define a layer to mirror the input
        Mirror = Lambda(mirror)
        # Define a layer to transpose the input
        Transpose = Lambda(transpose)
        # Rotating works by using that taking the mirror image of x and transposing items
        # is the same as rotating it by 90 degrees to the left (counterclockwise).
        
        # Define the same dueling convolutional network as before
        # Input is state
        sharedinput = Input(shape=self.state_size)
        # data_format="channels_first" so that input is (batch, channels, rows, cols) instead of (batch, rows, cols, channels)
        # Cond2D arguments: no of filters, size of the convolution window(here and next, (i,i) gives the same as (i)), strides over the picture
        # kernel 4x4, stride 1: 17x17 turns into 14x14
        shared = Conv2D(128, (4, 4), strides=(1), activation='relu', data_format="channels_first")(sharedinput)
        # kernel 4x4, stride 2: 14x14 turns into 6x6
        shared = Conv2D(64, (4, 4), strides=(2), activation='relu', data_format="channels_first")(shared)
        # kernel 2x2, stride 1: 6x6 turns into 5x5
        shared = Conv2D(64, (3, 3), strides=(1), activation='relu', data_format="channels_first")(shared)
        # Turn output from above into flat layer
        flatten = Flatten()(shared)
        
        # NN to predict the best action
        # Turn flattened above to 256
        advantage_fc = Dense(256, activation='relu')(flatten)
        # Map above to actions
        advantage = Dense(self.action_size, activation='linear')(advantage_fc)
        # We dont want the absolute advantage but the relative one, so we subtract the mean of all actions from every action
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), # keepdims=True so we get action_size dimension with the mean everywhere
                           output_shape=(self.action_size,))(advantage)
                           
        # NN to predict the value of the state
        # Turn flattened above to 256
        value_fc = Dense(256, activation='relu')(flatten)
        # Map above to one value, the value of state
        value =  Dense(1)(value_fc)
        # Expand this value into action_size dimensions to sum it with advantage
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(value)
        
        # Merge value and advantage into single output
        q_value = add([value, advantage])
        # Creates a model with .fit(), .predict() etc. from the layers.
        sharedq = Model(inputs=sharedinput, outputs=q_value)
        # This is now the model that will be used on every variation of the input.
        
        # Define the input
        input = Input(shape=self.state_size)
        # Mirror input
        inputM = Mirror(input)
        # Rotate input by 90 degrees to the left
        input90 = Transpose(inputM)
        # Mirror input90
        input90M = Mirror(input90)
        # Rotate input90 by 90 degrees to the left
        input180 = Transpose(input90M)
        # Mirror input180
        input180M = Mirror(input180)
        # Rotate input180 by 90 degrees to the left
        input270 = Transpose(input180M)
        # Mirror input270
        input270M = Mirror(input270)
        
        # List of layers, different versions of the input (mirrored, rotated)
        inputs = [input,inputM,input90,input90M,input180,input180M,input270,input270M]
        # List of the predictions for the inputs
        qinputs = [sharedq(inputs[i]) for i in range(8)]
        
        # Mapping of how the actions are transformed
        mapping = np.asarray([[0,1,3,2,1,0,2,3],[1,0,2,3,0,1,3,2],[2,2,0,0,3,3,1,1],[3,3,1,1,2,2,0,0],[4,4,4,4,4,4,4,4],[5,5,5,5,5,5,5,5]])
        
        # Average the predictions, while keeping in mind that the actions are not invariant under rotating and mirroring, then concatenating them back
        # [Lambda(lambda x: x[:,mapping[j,i]])(qinputs[i]) for i in range(8)] creates a list of the predictions for the different inputs for action j
        # Lambda(lambda x: K.expand_dims(x,-1))(Average()(....)) takes the average of those predictions and expands the last dimension to allow concatenating, (?)->(?,1)
        # Concatenate(axis=-1)([... for j in range(self.action_size)]) creates a list of the average predictions for every action and turns them into one array, (?,action_size)
        q_output = Concatenate(axis=-1)([Lambda(lambda x: K.expand_dims(x,-1))(Average()([Lambda(lambda x: x[:,mapping[j,i]])(qinputs[i]) for i in range(8)])) for j in range(self.action_size)])
        
        # Generate the final model
        model = Model(inputs=input, outputs=q_output)
        # Compile the model
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        
        return model
    
    elif version == "DUELCONVSYMI":
        # Here, I assume a state of shape (?,m,N,N) where the last two indices represent the board!
        
        # Define a matrix that will return the mirror image of the matrix
        MirrorMatrix = K.variable(np.rot90(np.identity(self.state_size[-1]),1))
        # Define a function to mirror a keras tensor
        def mirror(x):
            return K.dot(x,MirrorMatrix)
        # Define a function to transpose a keras tensor.
        def transpose(x):
            return K.permute_dimensions(x,(0,1,3,2))
        # Define a layer to mirror the input
        Mirror = Lambda(mirror)
        # Define a layer to transpose the input
        Transpose = Lambda(transpose)
        # Rotating works by using that taking the mirror image of x and transposing items
        # is the same as rotating it by 90 degrees to the left (counterclockwise).
        
        # Define the same dueling convolutional network as before
        # Input is state
        sharedinput = Input(shape=self.state_size)
        # data_format="channels_first" so that input is (batch, channels, rows, cols) instead of (batch, rows, cols, channels)
        # Cond2D arguments: no of filters, size of the convolution window(here and next, (i,i) gives the same as (i)), strides over the picture
        # kernel 4x4, stride 1: 17x17 turns into 13x13
        shared = Conv2D(128, (5, 5), strides=(1), activation='relu', data_format="channels_first")(sharedinput)
        # kernel 4x4, stride 2: 13x13 turns into 6x6
        shared = Conv2D(64, (3, 3), strides=(2), activation='relu', data_format="channels_first")(shared)
        # kernel 2x2, stride 1: 6x6 turns into 5x5
        shared = Conv2D(64, (3, 3), strides=(1), activation='relu', data_format="channels_first")(shared)
        # Turn output from above into flat layer
        flatten = Flatten()(shared)
        
        # NN to predict the best action
        # Turn flattened above to 256
        advantage_fc = Dense(256, activation='relu')(flatten)
        # Map above to actions
        advantage = Dense(self.action_size, activation='linear')(advantage_fc)
        # We dont want the absolute advantage but the relative one, so we subtract the mean of all actions from every action
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), # keepdims=True so we get action_size dimension with the mean everywhere
                           output_shape=(self.action_size,))(advantage)
                           
        # NN to predict the value of the state
        # Turn flattened above to 256
        value_fc = Dense(256, activation='relu')(flatten)
        # Map above to one value, the value of state
        value =  Dense(1)(value_fc)
        # Expand this value into action_size dimensions to sum it with advantage
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(value)
        # Merge value and advantage into single output
        q_value = add([value, advantage])
        # Creates a model with .fit(), .predict() etc. from the layers.
        sharedq = Model(inputs=sharedinput, outputs=q_value)
        # This is now the model that will be used on every variation of the input.
        
        # Define the input
        input = Input(shape=self.state_size)
        # Mirror input
        inputM = Mirror(input)
        # Rotate input by 90 degrees to the left
        input90 = Transpose(inputM)
        # Mirror input90
        input90M = Mirror(input90)
        # Rotate input90 by 90 degrees to the left
        input180 = Transpose(input90M)
        # Mirror input180
        input180M = Mirror(input180)
        # Rotate input180 by 90 degrees to the left
        input270 = Transpose(input180M)
        # Mirror input270
        input270M = Mirror(input270)
        
        # List of layers, different versions of the input (mirrored, rotated)
        inputs = [input,inputM,input90,input90M,input180,input180M,input270,input270M]
        # List of the predictions for the inputs
        qinputs = [sharedq(inputs[i]) for i in range(8)]
        
        # Mapping of how the actions are transformed
        mapping = np.asarray([[0,1,3,2,1,0,2,3],[1,0,2,3,0,1,3,2],[2,2,0,0,3,3,1,1],[3,3,1,1,2,2,0,0],[4,4,4,4,4,4,4,4],[5,5,5,5,5,5,5,5]])
        
        # Average the predictions, while keeping in mind that the actions are not invariant under rotating and mirroring, then concatenating them back
        # [Lambda(lambda x: x[:,mapping[j,i]])(qinputs[i]) for i in range(8)] creates a list of the predictions for the different inputs for action j
        # Lambda(lambda x: K.expand_dims(x,-1))(Average()(....)) takes the average of those predictions and expands the last dimension to allow concatenating, (?)->(?,1)
        # Concatenate(axis=-1)([... for j in range(self.action_size)]) creates a list of the average predictions for every action and turns them into one array, (?,action_size)
        q_output = Concatenate(axis=-1)([Lambda(lambda x: K.expand_dims(x,-1))(Average()([Lambda(lambda x: x[:,mapping[j,i]])(qinputs[i]) for i in range(8)])) for j in range(self.action_size)])
        
        # Generate the final model
        model = Model(inputs=input, outputs=q_output)
        # Compile the model
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        
        return model
    
    elif version == "DUELCONVSYMNOISE":
        # Here, I assume a state of shape (?,m,N,N) where the last two indices represent the board!
        
        # Define a matrix that will return the mirror image of the matrix
        MirrorMatrix = K.variable(np.rot90(np.identity(self.state_size[-1]),1))
        # Define a function to mirror a keras tensor
        def mirror(x):
            return K.dot(x,MirrorMatrix)
        # Define a function to transpose a keras tensor.
        def transpose(x):
            return K.permute_dimensions(x,(0,1,3,2))
        # Define a layer to mirror the input
        Mirror = Lambda(mirror)
        # Define a layer to transpose the input
        Transpose = Lambda(transpose)
        # Rotating works by using that taking the mirror image of x and transposing items
        # is the same as rotating it by 90 degrees to the left (counterclockwise).
        
        # Define the same dueling convolutional network as before
        # Input is state
        sharedinput = Input(shape=self.state_size)
        # data_format="channels_first" so that input is (batch, channels, rows, cols) instead of (batch, rows, cols, channels)
        # Cond2D arguments: no of filters, size of the convolution window(here and next, (i,i) gives the same as (i)), strides over the picture
        # kernel 4x4, stride 1: 17x17 turns into 14x14
        shared = Conv2D(128, (4, 4), strides=(1), activation='elu', data_format="channels_first")(sharedinput)
        # kernel 4x4, stride 2: 14x14 turns into 6x6
        shared = Conv2D(64, (4, 4), strides=(2), activation='elu', data_format="channels_first")(shared)
        # kernel 2x2, stride 1: 6x6 turns into 5x5
        shared = Conv2D(64, (3, 3), strides=(1), activation='elu', data_format="channels_first")(shared)
        # Turn output from above into flat layer
        flatten = Flatten()(shared)
        
        # NN to predict the best action
        # Turn flattened above to 256
        advantage_fc = NoisyDense(256, activation='elu')(flatten)
        # Map above to actions
        advantage = NoisyDense(self.action_size, activation='linear')(advantage_fc)
        # We dont want the absolute advantage but the relative one, so we subtract the mean of all actions from every action
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), # keepdims=True so we get action_size dimension with the mean everywhere
                           output_shape=(self.action_size,))(advantage)
                           
        # NN to predict the value of the state
        # Turn flattened above to 256
        value_fc = NoisyDense(256, activation='elu')(flatten)
        # Map above to one value, the value of state
        value =  NoisyDense(1)(value_fc)
        # Expand this value into action_size dimensions to sum it with advantage
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(value)
        # Merge value and advantage into single output
        q_value = add([value, advantage])
        # Creates a model with .fit(), .predict() etc. from the layers.
        sharedq = Model(inputs=sharedinput, outputs=q_value)
        # This is now the model that will be used on every variation of the input.
        
        # Define the input
        input = Input(shape=self.state_size)
        # Mirror input
        inputM = Mirror(input)
        # Rotate input by 90 degrees to the left
        input90 = Transpose(inputM)
        # Mirror input90
        input90M = Mirror(input90)
        # Rotate input90 by 90 degrees to the left
        input180 = Transpose(input90M)
        # Mirror input180
        input180M = Mirror(input180)
        # Rotate input180 by 90 degrees to the left
        input270 = Transpose(input180M)
        # Mirror input270
        input270M = Mirror(input270)
        
        # List of layers, different versions of the input (mirrored, rotated)
        inputs = [input,inputM,input90,input90M,input180,input180M,input270,input270M]
        # List of the predictions for the inputs
        qinputs = [sharedq(inputs[i]) for i in range(8)]
        
        # Mapping of how the actions are transformed
        mapping = np.asarray([[0,1,3,2,1,0,2,3],[1,0,2,3,0,1,3,2],[2,2,0,0,3,3,1,1],[3,3,1,1,2,2,0,0],[4,4,4,4,4,4,4,4],[5,5,5,5,5,5,5,5]])
        
        # Average the predictions, while keeping in mind that the actions are not invariant under rotating and mirroring, then concatenating them back
        # [Lambda(lambda x: x[:,mapping[j,i]])(qinputs[i]) for i in range(8)] creates a list of the predictions for the different inputs for action j
        # Lambda(lambda x: K.expand_dims(x,-1))(Average()(....)) takes the average of those predictions and expands the last dimension to allow concatenating, (?)->(?,1)
        # Concatenate(axis=-1)([... for j in range(self.action_size)]) creates a list of the average predictions for every action and turns them into one array, (?,action_size)
        q_output = Concatenate(axis=-1)([Lambda(lambda x: K.expand_dims(x,-1))(Average()([Lambda(lambda x: x[:,mapping[j,i]])(qinputs[i]) for i in range(8)])) for j in range(self.action_size)])
        
        # Generate the final model
        model = Model(inputs=input, outputs=q_output)
        # Compile the model
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        
        return model
    
    elif version == "DENSE":
        model = Sequential() # Linear stack of layers
        # Add layers
        model.add(Dense(512, input_dim=self.state_size, activation='elu')) # First layer with input
        model.add(Dense(512, activation='elu')) # Intermediate layer
        model.add(Dense(512, activation='elu')) # Intermediate layer
        model.add(Dense(self.action_size, activation='linear')) # Output layer, linear to the actions
        # Use optimizer Adam with mean squared loss
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        return model
    
    elif version == "DUELDENSE":
        input = Input(shape=self.state_size)
        # Add layers
        shared = Dense(512, input_dim=self.state_size, activation='elu')(input) # First layer with input
        shared = Dense(512, input_dim=self.state_size, activation='elu')(shared) # Intermediate layer
        
        # NN to predict the best action
        # Turn shared above to 256
        advantage_fc = Dense(256, activation='relu')(shared)
        # Map above to actions
        advantage = Dense(self.action_size, activation='linear')(advantage_fc)
        # We dont want the absolute advantage but the relative one, so we subtract the mean of all actions from every action
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), # keepdims=True so we get action_size dimension with the mean everywhere
                           output_shape=(self.action_size,))(advantage)
        
        # NN to predict the value of the state
        # Turn shared above to 256
        value_fc = Dense(256, activation='relu')(shared)
        # Map above to one value, the value of state
        value =  Dense(1)(value_fc)
        # Expand this value into action_size dimensions to sum it with advantage
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(value)
                       
        # Merge value and advantage into single output
        #q_value = merge([value, advantage], mode='sum')
        q_value = add([value, advantage])
        # Creates a model with .fit(), .predict() etc. from the layers.
        model = Model(inputs=input, outputs=q_value)
        # Compile the model
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        return model
    
    else:
        raise ValueError("There is no predefined model with this name!", version)


def replay(self, batch_size=16, update=False, PER=True):
    """ Replay random experiences and fit the model with them.
        They are either drawn from the SumTree or the deque.
    """
    # Either sample from experience for PER
    if PER:
        # Sample steps from our experience.
        tree_idx, minibatch, ISWeights_mb = self.experience.sample(batch_size)
        # Unpack batch
        old_state, action, reward, next_state, done = np.concatenate(minibatch[:,0,0],0), minibatch[:,0,1].astype(int), minibatch[:,0,2], np.concatenate(minibatch[:,0,3],0), minibatch[:,0,4].astype(bool)
    # Or sample from deque
    else:
        # Sample steps from our deque.
        minibatch = np.asarray(random.sample(self.training_data, batch_size))
        # Unpack batch
        old_state, action, reward, next_state, done = np.concatenate(minibatch[:,0],0), minibatch[:,1].astype(int), minibatch[:,2], np.concatenate(minibatch[:,3],0), minibatch[:,4].astype(bool)
    
    # Either use target network
    if self.use_target:
        # Predict the rewards for the next step with the target model
        Q_next = self.target_model.predict(next_state)
    # Or don't
    else:
        # Predict the rewards for the next step with the model
        Q_next = self.model.predict(next_state)
    
    # Predict the next action with the model
    next, values, possible = predict_action(self,next_state)
    # Calculate the reward of action plus the adjusted (with gamma) projected (target model) reward for the best next action.
    target = reward + self.gamma*Q_next[np.arange(next.shape[0]),next]
    # If the step terminated the round, omit the adjusted reward for the next step.
    target[done] = reward[done]
    # Predict the reward for the old state
    target_f = self.model.predict(old_state)
    if not self.abs_error_after_fit:
        target_f_copy = np.copy(target_f)
    # The reward for action should be the reward for the next action, plus the adjusted reward of the best action after that.
    target_f[np.arange(action.shape[0]),action] = target
    # Fit impossible actions as WAIT
    if self.fit_impossible_as_wait:
        target_f[np.invert(possible)] = [item for sublist in [[target_f[i,4]]*np.count_nonzero(np.invert(possible[i,:])) for i in range(batch_size)] for item in sublist]
    
    # For PER use weights in fit
    if PER:
        # Fit the model to return this as the reward of action at old_state.
        # Adjust how the loss is calculated with the IS weights, to allow countering the effects of PER.
        self.model.fit(old_state, target_f, epochs=1, verbose=0, sample_weight=ISWeights_mb)
    # Or not
    else:
        # Fit the model to return this as the reward of action at old_state.
        self.model.fit(old_state, target_f, epochs=1, verbose=0)
    
    # Only for PER update the priorities
    if PER:
        if self.abs_error_after_fit:
            # Calculate the absolute error of our prediction
            abs_errors = np.abs(target-self.model.predict(old_state)[np.arange(action.shape[0]),action]) # For some reason batch_update only accepts floats as weights.
        else:
            # Calculate the absolute error of our prediction
            abs_errors = np.abs(target-target_f_copy[np.arange(action.shape[0]),action]) # For some reason batch_update only accepts floats as weights.
        # Update experience tree with the absolute errors
        self.experience.batch_update(tree_idx, abs_errors)
    
    # After a round of training, update the target model?
    if update and self.use_target: update_target_model(self)
    
    # Counter for calculating the learning rate
    self.fitcount +=1


def adjust_epsilon(self):
    """ Adjust epsilon and epsilon_bomb.
        Uses exponential decay to the minimum.
        If epsilon is zero, it stays that way.
    """
    if self.adjust_epsilon and self.epsilon != 0 and self.epsilon > self.epsilon_min:
        self.epsilon = self.epsilon_min + (self.epsilon-self.epsilon_min)*np.exp(self.epsilon_decay)
    if self.adjust_epsilon and self.epsilon_bomb != 0 and self.epsilon_bomb > self.epsilon_bomb_min:
        self.epsilon_bomb = self.epsilon_bomb_min + (self.epsilon_bomb-self.epsilon_bomb_min)*np.exp(self.epsilon_bomb_decay)


def get_bools_remember(usePER=True, random=False):
    """ Get booleans for remember, depending on whether PER is used, training data is generated etc.
    """
    tree = usePER
    if random:
        tree = False
    deque = not usePER
    if random:
        deque = True
    return tree, deque


def save_config(self):
    """ Save configuration of the model etc., print and log it.
    """
    # General stuff, some things (forced random actions etc.) are still missing here!
    save_config = [(self.usePER,self.use_target,self.model_version,self.state_version,self.learning_bool_again,self.use_two_epsilon,self.force_random)]
    str1 = "PER: "+str(self.usePER)+", Target: "+str(self.use_target)+", Model: "+self.model_version+", State: "+self.state_version+", LoadModel: "+str(self.learning_bool_again)+", Use Two e: "+str(self.use_two_epsilon)+", Force: "+str(self.force_random)
    self.logger.info(str1)
    print(str1)
    config = str1
    # Learning parameters
    save_config += [(self.gamma,self.learning_rate,self.epsilon,self.epsilon_min,self.epsilon_decay,self.replay_batch_size,self.replay_batch_times,self.update_every,self.learning_rate_decay,self.start_training,self.train_every,self.random_actions_after_osc,
                        self.random_actions_after_wait,self.compare_diamond_size,self.epsilon_bomb,self.epsilon_bomb_min,self.epsilon_bomb_decay,self.no_wait_on_bomb,self.no_waiting_bomb_range,self.noimpossible,self.compare_diamond_bomb_size,
                        self.possible_future_state)]
    str2 = "Gamma: "+str(self.gamma)+", Learning Rate: "+str(self.learning_rate)+", LR_decay: 1/"+str(int(1/self.learning_rate_decay))+", e: "+str(self.epsilon)+", e_min: "+str(self.epsilon_min)+", e_decay: 1/"+str(int(1/self.epsilon_decay))
    self.logger.info(str2)
    config += "\n"+str2
    str21 = "e_bomb: "+str(self.epsilon_bomb)+", e_bomb_min: "+str(self.epsilon_bomb_min)+", e_bomb_decay: 1/"+str(int(1/self.epsilon_bomb_decay))
    self.logger.info(str21)
    config += "\n"+str21
    str3 = "Batch_size: "+str(self.replay_batch_size)+", Batch_times: "+str(self.replay_batch_times)+", Update Target every: "+str(self.update_every)+", Start_after: "+str(self.start_training)+", Train every: "+str(self.train_every)
    self.logger.info(str3)
    config += "\n"+str3
    str31 = "Random after Osc: "+str(self.random_actions_after_osc)+", Random after Wait: "+str(self.random_actions_after_wait)+", Diamond: "+str((self.compare_diamond_size,self.compare_diamond_bomb_size))+", No Wait on Bomb: "+str(self.no_wait_on_bomb)+", No Wait in Bomb range: "+str(self.no_waiting_bomb_range)+", No Impossible: "+str(self.noimpossible)
    self.logger.info(str31)
    config += "\n"+str31
    str32 = "Use Projected state: "+str(self.possible_future_state)
    self.logger.info(str32)
    config += "\n"+str32
    # Memory parameters
    save_config += [(self.experience_size,self.PER_max,self.PER_e,self.PER_a,self.PER_b,self.PER_b_increment)]
    save_config += [(self.training_data_size)]
    if self.usePER:
        str4 = "PER size: "+str(self.experience_size)+", PER_max: "+str(self.PER_max)+", PER_e: "+str(self.PER_e)+", PER_a: "+str(self.PER_a)+", PER_b: "+str(self.PER_b)+", PER_b_inc: "+str(self.PER_b_increment)
        self.logger.info(str4)
        config += "\n"+str4
    else:
        str4 = "Deque size: "+str(self.training_data_size)
        self.logger.info(str4)
        config += "\n"+str4
    # Rewards
    save_config += [self.rewards,self.value_useless_waiting,self.value_useless_bomb,self.value_wait_range_bomb,self.stuck_bomb]
    str5 = "Rewards: "+str(self.rewards)+", Useless: "+str((self.value_useless_waiting,self.value_useless_bomb,self.value_wait_range_bomb,self.stuck_bomb))
    self.logger.info(str5)
    config += "\n"+str5
    # Probabilities
    save_config += [self.pactions]
    str51 = "Probabilities: "+str(self.pactions)
    self.logger.info(str51)
    config += "\n"+str51
    # Settings of the environment
    str6 = str(s)
    self.logger.info(str6)
    config += "\n"+str6
    # Save it
    text_file = open(""+self.save_perf+"/config.txt", "w")
    text_file.write(config)
    text_file.close()
    np.save(""+self.save_perf+"/config.npy",save_config)
    np.save(""+self.save_perf+"/settings.npy",s)


def actions_possible(self,batch,noexplosions=True):
    """ Returns which actions are possible from this state (batch of states).
        Excludes actions the game deems impossible, along with walking into explosions since they guarantee death.
        The state is projected into the future, i.e. explosions fade and bombs explode, to determine this.
    """
    if self.noimpossible:
        if self.state_version == "ALLOH":
            # Get agent position
            pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
            # Get positions around agent in various dimensions
            shift=np.asarray([[0,0,-1,0],[0,0,1,0],[0,0,0,-1],[0,0,0,1]])
            if noexplosions:
                ind=np.repeat(np.expand_dims(np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2),0),4,0)
                ind[:,:,1,:]=np.asarray([0, 1, 5, 6])[:,np.newaxis,np.newaxis]
            else:
                ind=np.repeat(np.expand_dims(np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2),0),3,0)
                ind[:,:,1,:]=np.asarray([0, 1, 5])[:,np.newaxis,np.newaxis]
            # Use either projected state
            if self.possible_future_state:
                batch_exploded=explode_bombs(self,batch)
                possible=np.pad(np.logical_and.reduce([batch_exploded[tuple(zip(*index))]==0 for index in ind]),((0,0),(0,2)),"constant",constant_values=True)
            # or current state
            else:
                possible=np.pad(np.logical_and.reduce([batch[tuple(zip(*index))]==0 for index in ind]),((0,0),(0,2)),"constant",constant_values=True)
            # Using a bomb is only possible if the game says so
            pos[1,:]=4
            possible[:,5]=batch[tuple(pos)]==1
            # Waiting while standing on a bomb is stupid?
            # Dying by random chance when using bombs should be smaller with this, maybe exclude this once using bombs works.
            if self.no_wait_on_bomb:
                pos[1,:]=5
                possible[:,4]=np.logical_or(batch[tuple(pos)]==0,np.count_nonzero(possible,1)<2)
            # Do not wait while standing in range of a bomb
            if self.no_waiting_bomb_range:
                possible[np.logical_and(in_range_of_bomb(self, batch),np.count_nonzero(possible,1)>=2),4]=False
            return possible
        elif self.state_version == "ALLOHL":
            # Get agent position
            pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
            # Get positions around agent in various dimensions
            shift=np.asarray([[0,0,-1,0],[0,0,1,0],[0,0,0,-1],[0,0,0,1]])
            # Use either projected state
            if self.possible_future_state:
                if noexplosions:
                    ind=np.repeat(np.expand_dims(np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2),0),8,0)
                    ind[:,:,1,:]=np.asarray([0, 1, 5, 6, 7, 8, 9, 10])[:,np.newaxis,np.newaxis]
                else:
                    ind=np.repeat(np.expand_dims(np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2),0),6,0)
                    ind[:,:,1,:]=np.asarray([0, 1, 5, 6, 7, 8])[:,np.newaxis,np.newaxis]
                batch_exploded=explode_bombs(self,batch)
                possible=np.pad(np.logical_and.reduce([batch_exploded[tuple(zip(*index))]==0 for index in ind]),((0,0),(0,2)),"constant",constant_values=True)
            # or current state
            else:
                if noexplosions:
                    ind=np.repeat(np.expand_dims(np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2),0),7,0)
                    ind[:,:,1,:]=np.asarray([0, 1, 5, 6, 7, 8, 10])[:,np.newaxis,np.newaxis]
                else:
                    ind=np.repeat(np.expand_dims(np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2),0),6,0)
                    ind[:,:,1,:]=np.asarray([0, 1, 5, 6, 7, 8])[:,np.newaxis,np.newaxis]
                possible=np.pad(np.logical_and.reduce([batch[tuple(zip(*index))]==0 for index in ind]),((0,0),(0,2)),"constant",constant_values=True)
            # Using a bomb is only possible if the game says so
            pos[1,:]=4
            possible[:,5]=batch[tuple(pos)]==1
            # Waiting while standing on a bomb is stupid?
            # Dying by random chance when using bombs should be smaller with this, maybe exclude this once using bombs works.
            if self.no_wait_on_bomb:
                poswait = np.repeat(np.expand_dims(pos,0),1,0)
                poswait[:,1,:] = np.asarray([8])[:,np.newaxis] # only 8 is enough, agent cannot stand on a bomb that is not new if waiting is prohibited
                possible[:,4]=np.logical_or(np.logical_and.reduce([batch[tuple(pos)]==0 for pos in poswait]),np.count_nonzero(possible,1)<2)
            # Do not wait while standing in range of a bomb
            if self.no_waiting_bomb_range:
                possible[np.logical_and(in_range_of_bomb(self, batch),np.count_nonzero(possible,1)>=2),4]=False
            return possible
        elif self.state_version == "ALL":
            # Get agent position
            pos=np.asarray(np.where(batch[:,1:2,:,:]==1))
            # Get positions around agent in various dimensions
            shift=np.asarray([[0,0,-1,0],[0,0,1,0],[0,0,0,-1],[0,0,0,1]])
            if noexplosions:
                ind=np.repeat(np.expand_dims(np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2),0),3,0)
                ind[:,:,1,:]=np.asarray([0, 3, 4])[:,np.newaxis,np.newaxis]
            else:
                ind=np.repeat(np.expand_dims(np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2),0),2,0)
                ind[:,:,1,:]=np.asarray([0, 3])[:,np.newaxis,np.newaxis]
            # Walking is possible if neither walls/crates nor bombs or explosions are in the way
            # Excluding explosions is not strictly necessary as walking onto them is a valid action, it is just a stupid one in every possible situation
            # Use either projected state
            if self.possible_future_state:
                batch_exploded=explode_bombs(self,batch)
                possible=np.pad(np.logical_and.reduce([batch[tuple(zip(*index))]==0 for index in ind]),((0,0),(0,2)),"constant",constant_values=True)
            # or current state
            else:
                possible=np.pad(np.logical_and.reduce([batch_exploded[tuple(zip(*index))]==0 for index in ind]),((0,0),(0,2)),"constant",constant_values=True)
            # Using a bomb is only possible if the game says so
            pos[1,:]=2
            possible[:,5]=batch[tuple(pos)]==1
            # Waiting while standing on a bomb is stupid?
            # Dying by random chance when using bombs should be smaller with this, maybe exclude this once using bombs works.
            if self.no_wait_on_bomb:
                pos[1,:]=5
                possible[:,4]=np.logical_or(batch[tuple(pos)]==0,np.count_nonzero(possible,1)<2)
            # Do not wait while standing in range of a bomb
            if self.no_waiting_bomb_range:
                possible[np.logical_and(in_range_of_bomb(self, batch),np.count_nonzero(possible,1)>1),4]=False
            return possible
        elif self.state_version == "SMALL":
            # Get agent position
            pos=np.asarray(np.where(batch[:,1:2,:,:]==1))
            # Get positions around agent
            shift=np.asarray([[0,0,-1,0],[0,0,1,0],[0,0,0,-1],[0,0,0,1]])
            ind0=np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2)
            # Walking in every directions where there is a free tile is possible
            possible=np.pad(batch[tuple(zip(*ind0))]==0,((0,0),(0,2)),"constant",constant_values=True)
            return possible
        else:
            raise ValueError("There is no predefined model with this name!", version)
    # Can also return just True, this makes all actions possible
    else:
        return np.full((batch.shape[0],6),True)


def predict_action(self,batch, first=False):
    """ Predict the best actions with the model, take the best action that is possible from this state.
        Also returns the Q values so that they can be logged.
    """
    # If only possible actions should be considered
    if self.noimpossible:
        # Get possible actions
        possible = actions_possible(self,batch,self.explosions_impossible)[:,:self.action_size]
        # Not Bomb as first action
        if first:
            possible[0,5] = False
        # Predict Q values for batch
        predict = self.model.predict(batch)
        values = np.copy(predict)
        # Set the Q values for the impossible actions to -infinity, so that they will not be chosen in argmax
        predict[np.invert(possible)] = -np.inf
        # Choose the actions with the largest predicted Q value
        return np.argmax(predict,1), values, possible
    # Normal version
    else:
        # Predict Q values for batch
        values = self.model.predict(batch)
        # Choose the actions with the largest predicted Q value
        return np.argmax(values,1), values, np.full((batch.shape[0],6),True)


def states_equal(self,new_state,old_state,proximity=True):
    """ Compares whether the proximity of the agent is the same for both states.
    """
    if proximity:
        if self.state_version == "ALLOH":
            # Get agent position
            pos=np.asarray(np.where(new_state[:,3:4,:,:]==1))
            pos_old=np.asarray(np.where(old_state[:,3:4,:,:]==1))
        elif self.state_version == "ALLOHL":
            # Get agent position
            pos=np.asarray(np.where(new_state[:,3:4,:,:]==1))
            pos_old=np.asarray(np.where(old_state[:,3:4,:,:]==1))
        elif self.state_version == "ALL":
            # Get agent position
            pos=np.asarray(np.where(new_state[:,1:2,:,:]==1))
            pos_old=np.asarray(np.where(old_state[:,1:2,:,:]==1))
        elif self.state_version == "SMALL":
            # Get agent position
            pos=np.asarray(np.where(new_state[:,1:2,:,:]==1))
            pos_old=np.asarray(np.where(old_state[:,1:2,:,:]==1))
        else:
            raise ValueError("There is no predefined model with this name!", version)
        # Size of the diamond
        n = self.compare_diamond_size
        # Get indices around position
        index = np.indices((2*n+1,2*n+1)) + np.expand_dims(pos[2:]-n,1)
        index[index<0] = 0
        index[index>16] = 16
        # Create diamond as a boolean array
        grid = np.abs(np.indices((2*n+1,2*n+1)) - np.expand_dims(np.expand_dims([n,n],1),1))
        diamond = grid[0] + grid[1] <= n
        # Compare only slices of the original arrays
        equal = np.array_equal(new_state[:,:,index[0],index[1]][:,:,diamond],old_state[:,:,index[0],index[1]][:,:,diamond])
        # If the slices are equal, and the position is the same, the action did not have a useful effect
        return equal and np.array_equal(pos,pos_old)
    else:
        return np.array_equal(new_state,old_state)


def is_bomb_near(self,batch):
    """ Checks whether a bomb is ticking in the proximity of self.
    """
    if self.state_version == "ALLOH":
        # Get agent position
        pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
    elif self.state_version == "ALLOHL":
        # Get agent position
        pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
    elif self.state_version == "ALL":
        # Get agent position
        pos=np.asarray(np.where(batch[:,1:2,:,:]==1))
    elif self.state_version == "SMALL":
        # SMALL cannot contain bombs, just return False
        return np.full((batch.shape[0]),False)
    else:
        raise ValueError("There is no predefined model with this name!", version)
    # Size of the diamond
    n = self.compare_diamond_bomb_size
    # Get indices around position
    index = np.tile(np.expand_dims(np.indices((2*n+1,2*n+1)),3),(1,1,1,pos.shape[1])) + np.expand_dims(np.expand_dims(pos[2:,:]-np.expand_dims([n,n],1),1),1)
    index[index<0] = 0
    index[index>16] = 16
    # Create diamond as a boolean array
    grid = np.abs(np.indices((2*n+1,2*n+1)) - np.expand_dims(np.expand_dims([n,n],1),1))
    diamond = grid[0] + grid[1] <= n
    if self.state_version == "ALLOH":
        # Check whether there are bombs in diamond
        return np.any(batch[pos[0],5,index[0],index[1]][diamond,:],0)
    elif self.state_version == "ALLOHL":
        # Check whether there are bombs in diamond
        return np.logical_or.reduce(np.any(batch[pos[0],5:9,index[0],index[1]][diamond,:],0),1)
    elif self.state_version == "ALL":
        # Check whether there are bombs in diamond
        return np.any(batch[pos[0],3,index[0],index[1]][diamond,:],0)
    else:
        raise ValueError("There is no predefined model with this name!", version)


def in_range_of_bomb(self,batch):
    """ Checks whether agent is in the explosion range of a bomb.
    """
    if self.state_version == "ALLOH":
        # Get agent position
        pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
    elif self.state_version == "ALLOHL":
        # Get agent position
        pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
    elif self.state_version == "ALL":
        # Get agent position
        pos=np.asarray(np.where(batch[:,1:2,:,:]==1))
    elif self.state_version == "SMALL":
        # SMALL cannot contain bombs, just return False
        return np.full((batch.shape[0]),False)
    else:
        raise ValueError("There is no predefined model with this name!", version)
    # Size of the cross
    n = self.bomb_power
    # Get indices around position
    index = np.tile(np.expand_dims(np.indices((2*n+1,2*n+1)),3),(1,1,1,pos.shape[1])) + np.expand_dims(np.expand_dims(pos[2:,:]-np.expand_dims([n,n],1),1),1)
    index[index<0] = 0
    index[index>16] = 16
    # Create cross as a boolean array
    grid = np.abs(np.indices((2*n+1,2*n+1)) - np.expand_dims(np.expand_dims([n,n],1),1))
    cross = np.repeat(np.expand_dims(np.logical_or(grid[0]==0,grid[1]==0),0),pos.shape[1],0)
    shift=np.asarray([[0,0,-1,0],[0,0,1,0],[0,0,0,-1],[0,0,0,1]])
    ind=np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2)
    cross[:,1,n], cross[:,-2,n], cross[:,n,1], cross[:,n,-2] = zip(*(batch[tuple(zip(*ind))]==0))
    if self.state_version == "ALLOH":
        # Check whether there are bombs in cross
        return np.asarray([np.any(batch[pos[0,inde],5,index[0,cro,inde],index[1,cro,inde]]) for inde, cro in enumerate(cross)])
    elif self.state_version == "ALLOHL":
        # Check whether there are bombs in cross
        return np.asarray([np.any(batch[pos[0,inde],5:9,index[0,cro,inde],index[1,cro,inde]]) for inde, cro in enumerate(cross)])
    elif self.state_version == "ALL":
        # Check whether there are bombs in cross
        return np.asarray([np.any(batch[pos[0,inde],3,index[0,cro,inde],index[1,cro,inde]]) for inde, cro in enumerate(cross)])
    else:
        raise ValueError("There is no predefined model with this name!", version)


def save_sumtree(self):
    """ Save the contents of memory. Works in theory, but takes a huge amount of RAM!
    """
    np.savez_compressed("memory/"+self.save_as+"_memoryN"+str(self.counter)+".npz",
                        capacity=self.experience.SumTree.capacity, tree=self.experience.SumTree.tree, data=self.experience.SumTree.data)


def load_sumtree(self):
    """ Load saved memory.
    """
    experience = np.load("memory/"+self.load_memory+".npz")
    self.experience = Memory(memory["capacity"], self)
    self.experience.SumTree.tree = memory["tree"]
    self.experience.SumTree.data = memory["data"]


def first_round_reset(self,state):
    """ Reset memory at the beginning of each round.
    """
    self.memory["NEW_STATE"] = state
    self.memory["NEW_RANDOM"] = True
    self.last_events = deque(maxlen=self.max_loop)
    self.last_events_random = deque(maxlen=self.max_loop)
    self.last_events.appendleft(self.memory["NEW_STATE"])
    self.last_events_random.appendleft(True)
    self.memory["OLD_STATE"] = np.full((1,)+self.state_size,0)
    self.memory["OLD_RANDOM"] = True
    self.memory["OLD_OLD_STATE"] = np.full((1,)+self.state_size,1)
    self.memory["OLD_OLD_RANDOM"] = True
    self.memory["LAST_ACTION_RANDOM"] = True
    self.memory["USELESS_WAIT"] = False
    self.memory["USELESS_OSC"] = False
    self.memory["LOOP_DETECTED"] = False
    self.memory["USELESS_WAIT_REWARD"] = False
    self.memory["USELESS_OSC_REWARD"] = False
    self.memory["LOOP"] = False
    self.memory["RANDOM"] = 0
    self.memory["LAST_ACTION"] = 4
    self.memory["ACTION"] = 4


def explode_bombs(self,batch):
    """ Add explosion for all bombs that are about to explode, destroy crates and fade old explosions. ONLY FOR BOMB_POWER=3!
    """
    if self.state_version == "ALLOH":
        # Get bomb positions
        pos=np.asarray(np.where(batch[:,5:6,:,:]==1))
    elif self.state_version == "ALLOHL":
        # Get bomb positions
        pos=np.asarray(np.where(batch[:,5:6,:,:]==1))
    elif self.state_version == "ALL":
        # Get bomb positions
        pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
    elif self.state_version == "SMALL":
        # SMALL cannot contain bombs, just return the array again
        return batch
    else:
        raise ValueError("There is no predefined model with this name!", version)
    # Copy batch to avoid changing the original arrays?
    batch_copy = np.copy(batch)
    # Return batch with faded explosions when there are no bombs!
    if pos.shape[1]==0:
        if self.state_version == "ALLOH":
            # Old explosions fade
            batch_copy[:,6,:,:][batch_copy[:,6,:,:]!=0]-=1
            return batch_copy
        elif self.state_version == "ALLOHL":
            # Old explosions fade
            batch_copy[:,9,:,:]=batch_copy[:,10,:,:]
            batch_copy[:,10,:,:]=0
            return batch_copy
        elif self.state_version == "ALL":
            # Old explosions fade
            batch_copy[:,4,:,:][batch_copy[:,4,:,:]!=0]-=1
            return batch_copy
        else:
            raise ValueError("There is no predefined model with this name!", version)
    # Size of the cross
    n = self.bomb_power
    # Get indices around position
    index = np.tile(np.expand_dims(np.indices((2*n+1,2*n+1)),3),(1,1,1,pos.shape[1])) + np.expand_dims(np.expand_dims(pos[2:,:]-np.expand_dims([n,n],1),1),1)
    index[index<0] = 0
    index[index>16] = 16
    # Create cross as a boolean array
    grid = np.abs(np.indices((2*n+1,2*n+1)) - np.expand_dims(np.expand_dims([n,n],1),1))
    cross = np.repeat(np.expand_dims(np.logical_or(grid[0]==0,grid[1]==0),0),pos.shape[1],0)
    shift=np.asarray([[0,0,-1,0],[0,0,1,0],[0,0,0,-1],[0,0,0,1]])
    ind=np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2)
    cross[:,1,n], cross[:,-2,n], cross[:,n,1], cross[:,n,-2] = zip(*(batch[tuple(zip(*ind))]==0))
    if self.state_version == "ALLOH":
        # Old explosions fade
        batch_copy[:,6,:,:][batch_copy[:,6,:,:]!=0]-=1
        # Add explosions for bombs and destroy crates
        for inde, cro in enumerate(cross):
            batch_copy[pos[0,inde],6,index[0,cro,inde],index[1,cro,inde]]=2
            batch_copy[pos[0,inde],1,index[0,cro,inde],index[1,cro,inde]]=0
        return batch_copy
    elif self.state_version == "ALLOHL":
        # Old explosions fade
        batch_copy[:,9,:,:]=batch_copy[:,10,:,:]
        batch_copy[:,10,:,:]=0
        # Add explosions for bombs and destroy crates
        for inde, cro in enumerate(cross):
            batch_copy[pos[0,inde],10,index[0,cro,inde],index[1,cro,inde]]=1
            batch_copy[pos[0,inde],1,index[0,cro,inde],index[1,cro,inde]]=0
        return batch_copy
    elif self.state_version == "ALL":
        # Old explosions fade
        batch_copy[:,4,:,:][batch_copy[:,4,:,:]!=0]-=1
        # Add explosions for bombs and destroy crates
        for inde, cro in enumerate(cross):
            batch_copy[pos[0,inde],4,index[0,cro,inde],index[1,cro,inde]]=2
            batch_copy[pos[0,inde],0,index[0,cro,inde],index[1,cro,inde]][batch[pos[0,inde],0,index[0,cro,inde],index[1,cro,inde]]==1]=0
        return batch_copy
    else:
        raise ValueError("There is no predefined model with this name!", version)


class SumTree(object):
    """ SumTree to efficiently store our experiences with a priority.
    """
    
    # Index of the last element stored in our tree, we fill it up from left to right.
    last = 0
    
    def __init__(self, capacity):
        # Capacity of the tree
        self.capacity = capacity
        # Priorities
        self.tree = np.zeros(2*capacity - 1)
        # Stored elements
        self.data = np.zeros(capacity, dtype=object)
    
    def add(self, priority, data):
        """ Add a new element to the tree.
        """
        
        # Get the first free index
        index = self.last + self.capacity - 1
        # Add the data
        self.data[self.last] = data
        # Propagate the priority upwards
        self.propagate(index, priority)
        # Shift the pointer to the next free entry
        self.last += 1
        # Start overwriting from the left if the tree is full
        if self.last >= self.capacity:
            self.last = 0
            
    def propagate(self, index, priority):
        """ Propagate the new priority of a leaf upwards.
        """
        
        # Avoid .
        tree = self.tree
        # Get the difference between priorities
        change = priority - tree[index]
        # Assign the leaf the new priority
        tree[index] = priority
        # Then propagate the change through tree
        while index != 0:
            # Go one node upwards
            index = (index - 1)//2
            # Add the change to the priority of the node.
            tree[index] += change
    
    def get(self, priority):
        """ Get a leaf.
            
            Assume that the priority here comes from a uniform distribution between zero and the total priority.
            Then if it is smaller than the left_child priority, it is from a uniform distribution between zero and
            the total priority of the left subtree. If it is larger, priority minus the total priority of the left
            subtree is from a uniform distribution between zero and the total priority of the right subtree. Then
            then the initial situation is the same as before, but one leaf farther down the tree.
            In summary, the probability of one subtree to be chosen over the other one is directly proportional to
            its total priority. Propagating like this through the whole tree, the probability for choosing an element
            is this element's priority over the total priority of the tree.
            
            However, we divide the total priority into ranges and sample priority uniformly from them. This does not
            (in the limit of a lot of sampling) affect the probybility of elements to be chosen, but it forces sampling
            from every part of the Tree. Effectively, the Tree is divided into SubTrees of equal priority, and an element
            is drawn from each of them. This ensures that our batches are not correlated. Since new steps are added with
            the maximal priority, new elements would otherwise be quite likely to be sampled together!
        """
        
        # Avoid . and repetition
        tree = self.tree
        size = len(tree)
        # Start at the top
        parent = 0
        # Go down
        while True:
            # Left child of parent
            left_child = 2*parent + 1
            # Right child of parent
            right_child = left_child + 1
            # Stop if parent does not have any children
            if left_child >= size:
                leaf = parent
                break
            # Otherwise, continue downwards
            else:
                # Go either left
                if priority <= tree[left_child]:
                    parent = left_child
                # Or right
                else:
                    # If right, adjust the priority
                    priority -= tree[left_child]
                    parent = right_child
        # Index of the data corresponding to leaf
        data = leaf - self.capacity + 1
        # Return index, priority, data
        return leaf, tree[leaf], self.data[data]
    
    @property
    def total_priority(self):
        """ Get the total priority of the tree.
        """
        
        return self.tree[0]

class Memory(object):
    """ Our memory, it is dependent on SumTree.
    """
    
    def __init__(self, capacity, agent):
        # Hyperparameters for our sampling, get them from agent.
        self.PER_max = agent.PER_max
            # Maximal priority for a leaf.
        self.PER_e = agent.PER_e
            # Minimal priority for a leaf.
        self.PER_a = agent.PER_a
            # Tradeoff between randomness and prioritized sampling
        self.PER_b = agent.PER_b
            # Amount of randomness, 1 completely random
        self.PER_b_increment = agent.PER_b_increment
            # PER_b gets raised by this amount every time adjust_b is called
        
        self.agent = agent
        
        self.SumTree = SumTree(capacity)
    
    def store(self, elem):
        """ Store a new step in our memory.
            We add it with the maximal priority of any element in our tree.
        """
        
        # Get the max priority in the tree
        max_priority = np.max(self.SumTree.tree[-self.SumTree.capacity:])
        # If this is still zero, use the upper bound as priority
        if max_priority == 0:
            max_priority = self.PER_max**self.PER_a
        # Add the step to the tree
        self.SumTree.add(max_priority, elem)
    
    def sample(self, n):
        """ Get a sample of steps.
            
            We would like to be able to alleviate the effect of non-random sampling over time, since this otherwise
            endangers the convergence of our algorithm. So we introduce importance sampling weights, which will be used
            when fitting with our batch to adjust the relative weight of the samples. The loss the optimizer will try to
            minimize is then not the sum of the square loss, but the sum over the square loss of each sample weighted with ISWeights.
            For PER_b=0, all weights are simply one and do not affect the fit. For PER_b=1, the weights will be
            proportional to the sampling probability, thereby exactly countering the effect of not sampling randomly.
            We raise PER_b over the course of training, to achieve faster improvements in the beginning while not
            spoiling convergence in the end.
            
            We divide the total priority into segments to reduce correlations between the samples.
        """
        # Create an array to store the samples, use list for speed and convert to array later
        memory_b = []
        # Initialize the position of the samples and the weights later used for fitting
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n), dtype=np.float32)
        
        # Divide the total priority into n ranges to sample from
        segment = self.SumTree.total_priority/n
    
        # Every time a batch is sampled, PER_b is increased, till it reaches 1
        self.PER_b = np.min([1.,self.PER_b+self.PER_b_increment])
        # The adjusted value is returned to our agent to allow tracking it
        self.agent.PER_b = self.PER_b
        
        # The max weight is calculated
        p_min = np.max((np.min(self.SumTree.tree[-self.SumTree.capacity:]),self.PER_e**self.PER_a))/self.SumTree.total_priority
            # np.max is necessary when the SumTree is not completely filled
        max_weight = (p_min*n)**(-self.PER_b)
        
        # For every segment, a sample is drawn
        for i in range(n):
            # Random value in our range
            value = np.random.uniform(segment*i, segment*(i+1))
            # A sample is drawn
            index, priority, data = self.SumTree.get(value)
            
            # The probability to be sampled is proportional to the priority
            sampling_probability = priority/self.SumTree.total_priority
            
            #  The weight for fitting is calculated and stored
            b_ISWeights[i] = np.power(n*sampling_probability,-self.PER_b)/max_weight
            # The index is stored
            b_idx[i]= index
            # The step is stored
            memory_b.append([data])
        
        return b_idx, np.asarray(memory_b), b_ISWeights
    
    def batch_update(self, idx, errors):
        """ Update the priorities of a batch. The errors have to be positive!
            
            We want the elements to have varying priorities, however we do not want the priorities to play too
            large a role. So we take the errors to the power of PER_a. For PER_a=1 this just returns the original
            errors as priorities, for PER_a=0 this results in the same prioritiy for every error. PER_a can be
            adjusted to strike a suitable balance.
            Additionally, we do not want elements with priority zero, since they would never be sampled. So we
            add a minimum priority to every error.
            Also, the absolute errors are capped at PER_max, so that the influence one element can have is limited.
        """
        # Avoid zero priority
        errors += self.PER_e
        # Avoid a priority larger than the max priority
        clipped_errors = np.minimum(errors, self.PER_max)
        # Calculate the priorities
        ps = np.power(clipped_errors, self.PER_a)
        # Propagate the new priorities through the tree
        for ti, p in zip(idx, ps):
            self.SumTree.propagate(ti, p)
        
class NoisyDense(Layer):
    """ Modification of the Dense layer from keras. My own implementation of a noisy dense layer.
        
        Creates two weights, multiplies one of them with random numbers and sums them up.
        Everything else is like Dense, except I deleted some things I do not use for clarity.
        A lot slower because the numbers have to be generated, maybe use factorized noise or change how it is generated?
    """
    
    def __init__(self, units,
                 stddev=0.5,
                 initial_weights=0.017,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.stddev = stddev
        self.initial_weights = initial_weights
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer_n = initializers.Constant(value=self.initial_weights)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        
        # Normal weights
        weights = self.add_weight(shape=(input_dim, self.units), initializer=self.kernel_initializer, name='weights',)
        # Weights for random numbers, initialized with a constant
        weights_n = self.add_weight(shape=(input_dim, self.units), initializer=self.kernel_initializer_n, name='weights_n',)
        # The kernel is both weights, weights_n get multiplied by random numbers
        self.kernel = weights + weights_n * K.random_normal(shape=(input_dim, self.units), mean=0, stddev=self.stddev)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
    
    def get_config(self):
        config = {
            'units': self.units,
            'stddev': self.stddev,
            'initial_weights': self.initial_weights,
            'activation': activations.serialize(self.activation)
        }
        base_config = super(NoisyDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def populate_experience(self, data, usePER=True):
    """ Fill the memory buffer with data.
        Used to prepopulate our memory with saved data.
    """
    if usePER:
        for elem in data:
            self.experience.store(elem)
    else:
        for elem in data:
            self.training_data.append(elem)

def next_batch(self,batch,actions):
    """ Explosions fade, bomb counters go down, player moves and plants bombs. Bombs explode, crates are destroyed.
        There must be as many actions as states.
    """
    # Copy batch to avoid changing the original arrays?
    batch_copy = np.copy(batch)
    if self.state_version == "ALLOH":
        # Get bomb positions
        pos=np.asarray(np.where(batch[:,5:6,:,:]==1))
        # Get agent position
        agent_pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
    elif self.state_version == "ALLOHL":
        # Get bomb positions
        pos=np.asarray(np.where(batch[:,5:6,:,:]==1))
        # Get agent position
        agent_pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
    elif self.state_version == "ALL":
        # Get bomb positions
        pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
        # Get agent position
        agent_pos=np.asarray(np.where(batch[:,1:2,:,:]==1))
    elif self.state_version == "SMALL":
        # Get agent position
        agent_pos=np.asarray(np.where(batch[:,1:2,:,:]==1))
        # SMALL cannot contain bombs, just shift the agent.
        next_pos=agent_pos+np.squeeze(np.transpose([self.shift_dict[act] for act in action]),0)
        batch_copy[agent_pos[0],1,agent_pos[2],agent_pos[3]]=0
        batch_copy[next_pos[0],1,next_pos[2],next_pos[3]]=1
        return batch
    else:
        raise ValueError("There is no predefined model with this name!", version)
    # Fade explosions, bomb counters, shift player and add bombs
    if self.state_version == "ALLOH":
        # Old explosions fade
        batch_copy[:,6,:,:][batch_copy[:,6,:,:]!=0]-=1
        # Bomb counters run down
        batch_copy[:,5,:,:][batch_copy[:,5,:,:]!=0]-=1
        # Shift player according to action
        next_pos=agent_pos+np.squeeze(np.transpose([self.shift_dict[act] for act in actions]),0)
        batch_copy[agent_pos[0],3,agent_pos[2],agent_pos[3]]=0
        batch_copy[next_pos[0],3,next_pos[2],next_pos[3]]=1
        # Add bombs if action is bomb
        batch_copy[agent_pos[0][actions==5],5,agent_pos[2][actions==5],agent_pos[3][actions==5]]=4
    elif self.state_version == "ALLOHL":
        # Old explosions fade
        batch_copy[:,9,:,:]=batch_copy[:,10,:,:]
        batch_copy[:,10,:,:]=0
        # Old bomb counters run down
        batch_copy[:,5,:,:]=batch_copy[:,6,:,:]
        batch_copy[:,6,:,:]=batch_copy[:,7,:,:]
        batch_copy[:,7,:,:]=batch_copy[:,8,:,:]
        batch_copy[:,8,:,:]=0
        # Shift player according to action
        next_pos=agent_pos+np.squeeze(np.transpose([self.shift_dict[act] for act in actions]),0)
        batch_copy[agent_pos[0],3,agent_pos[2],agent_pos[3]]=0
        batch_copy[next_pos[0],3,next_pos[2],next_pos[3]]=1
        # Add bombs if action is bomb
        batch_copy[agent_pos[0][actions==5],8,agent_pos[2][actions==5],agent_pos[3][actions==5]]=1
    elif self.state_version == "ALL":
        # Old explosions fade
        batch_copy[:,4,:,:][batch_copy[:,4,:,:]!=0]-=1
        # Bomb counters run down
        batch_copy[:,3,:,:][batch_copy[:,3,:,:]!=0]-=1
        # Shift player according to action
        next_pos=agent_pos+np.squeeze(np.transpose([self.shift_dict[act] for act in actions]),0)
        batch_copy[agent_pos[0],1,agent_pos[2],agent_pos[3]]=0
        batch_copy[next_pos[0],1,next_pos[2],next_pos[3]]=1
        # Add bombs if action is bomb
        batch_copy[agent_pos[0][actions==5],3,agent_pos[2][actions==5],agent_pos[3][actions==5]]=4
    else:
        raise ValueError("There is no predefined model with this name!", version)
    # Explode bombs and destroy crates, but only if there are bombs
    if pos.shape[1]!=0:
        # Size of the cross
        n = self.bomb_power
        # Get indices around position
        index = np.tile(np.expand_dims(np.indices((2*n+1,2*n+1)),3),(1,1,1,pos.shape[1])) + np.expand_dims(np.expand_dims(pos[2:,:]-np.expand_dims([n,n],1),1),1)
        index[index<0] = 0
        index[index>16] = 16
        # Create cross as a boolean array
        grid = np.abs(np.indices((2*n+1,2*n+1)) - np.expand_dims(np.expand_dims([n,n],1),1))
        cross = np.repeat(np.expand_dims(np.logical_or(grid[0]==0,grid[1]==0),0),pos.shape[1],0)
        shift=np.asarray([[0,0,-1,0],[0,0,1,0],[0,0,0,-1],[0,0,0,1]])
        ind=np.expand_dims(shift.T,0)+np.expand_dims(pos.T,2)
        cross[:,1,n], cross[:,-2,n], cross[:,n,1], cross[:,n,-2] = zip(*(batch[tuple(zip(*ind))]==0))
        if self.state_version == "ALLOH":
            # Add explosions for bombs and destroy crates
            for inde, cro in enumerate(cross):
                batch_copy[pos[0,inde],6,index[0,cro,inde],index[1,cro,inde]]=2
                batch_copy[pos[0,inde],1,index[0,cro,inde],index[1,cro,inde]]=0
            return batch_copy
        elif self.state_version == "ALLOHL":
            # Add explosions for bombs and destroy crates
            for inde, cro in enumerate(cross):
                batch_copy[pos[0,inde],10,index[0,cro,inde],index[1,cro,inde]]=1
                batch_copy[pos[0,inde],1,index[0,cro,inde],index[1,cro,inde]]=0
            return batch_copy
        elif self.state_version == "ALL":
            # Add explosions for bombs and destroy crates
            for inde, cro in enumerate(cross):
                batch_copy[pos[0,inde],4,index[0,cro,inde],index[1,cro,inde]]=2
                batch_copy[pos[0,inde],0,index[0,cro,inde],index[1,cro,inde]][batch[pos[0,inde],0,index[0,cro,inde],index[1,cro,inde]]==1]=0
            return batch_copy
        else:
            raise ValueError("There is no predefined model with this name!", version)
    # Otherwise return
    else:
        return batch_copy

def is_terminal(self,batch):
    """ Checks whether the states are terminal (standing on explosion).
    """
    if self.state_version == "ALLOH":
        # Get agent position
        pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
        # Terminal if standing on explosion
        terminal = batch[pos[0],6,pos[2],pos[3]]==0
        return terminal
    elif self.state_version == "ALLOHL":
        # Get agent position
        pos=np.asarray(np.where(batch[:,3:4,:,:]==1))
        # Terminal if standing on explosion
        terminal = batch[pos[0],9:11,pos[2],pos[3]]==0
        return np.logical_and.reduce(terminal,1)
    elif self.state_version == "ALL":
        # Get agent position
        pos=np.asarray(np.where(batch[:,1:2,:,:]==1))
        # Terminal if standing on explosion
        terminal = batch[pos[0],4,pos[2],pos[3]]==0
        return terminal
    elif self.state_version == "SMALL":
        # SMALL cannot contain bombs, just return False
        return np.full((batch.shape[0]),False)
    else:
        raise ValueError("There is no predefined model with this name!", version)

def rotate_elem(self, elem):
    """ Just use a model that exploits this symmetry!
        I only kept this function in case I need it later.
        
        Rotate a step by 90 degrees to the right.
        Only works for states of shape (a, b, N, N, done).
        
        Also possible to rotate more often here, with np.rot90(...,N,...) and
        for i in range(N): action = self.actions_rotate90_dict[action]
        But huge overhead, so an order of magnitude faster to just iterate
        and save in between.
    """
    # Unpack element
    old_state, action, reward, next_state, done = elem
    # Check whether state has the right shape to be rotated!
    assert len(old_state.shape)==4
    # Rotate states
    old_state = np.rot90(old_state,1,(3,2))
    next_state = np.rot90(next_state,1,(3,2))
    # Rotate action
    action = self.actions_rotate90_dict[action]
    # Return as tuple
    return (old_state, action, reward, next_state, done)
    
    


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def simple_setup(self):
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def choose_simple(self):
    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x,y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x,y))

    # Check which moves make sense at all
    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
            (self.game_state['explosions'][d] <= 1) and
            (bomb_map[d] > 0) and
            (not d in others) and
            (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x-1,y) in valid_tiles: valid_actions.append('LEFT')
    if (x+1,y) in valid_tiles: valid_actions.append('RIGHT')
    if (x,y-1) in valid_tiles: valid_actions.append('UP')
    if (x,y+1) in valid_tiles: valid_actions.append('DOWN')
    if (x,y)   in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x,y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x,y), targets, self.logger)
    if d == (x,y-1): action_ideas.append('UP')
    if d == (x,y+1): action_ideas.append('DOWN')
    if d == (x-1,y): action_ideas.append('LEFT')
    if d == (x+1,y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x,y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x,y) and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for xb,yb,t in bombs:
        if (xb == x) and (abs(yb-y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb-x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for xb,yb,t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            self.next_action = a
            break

    # Keep track of chosen action for cycle detection
    if self.next_action == 'BOMB':
        self.bomb_history.append((x,y))