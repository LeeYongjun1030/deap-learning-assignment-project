# Environment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

from environment import Simple_Maze
import random
from keras.layers import *
from keras.models import *
from keras.optimizers import *
print("Loaded")

np.random.seed(1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def basicANN(data, n):
    data = Dense(units=n)(data)
    data = BatchNormalization()(data)
    data = Activation('tanh')(data)
    
    return data

def simpleNet(x):
  
    x = basicANN(x,16)
    x = basicANN(x,32)
    x = Dense(units=4, activation='linear')(x)
    
    return x

class Agent:
    def __init__(self, environment):
        
        # Q value estimator
        self.input_data = Input(shape=(8,))
        self.output_data = simpleNet(self.input_data)
        self.estimator = Model(inputs=self.input_data, outputs = self.output_data)
        self.optimizer = Adam(lr=0.001)
        self.estimator.compile(self.optimizer, 'mse')

        self.environment = environment
        
        # Hyper parameters
        self.discount_reward = 0.9
        self.epsilon = 0.8
        self.discount_epsilon = 0.9999
        self.memory_size = 1000
        self.save_ratio = 1
        self.train_ratio = 0.1
        self.lr = 0.001
        self.epsilon_min = 0.05
        self.step = 0
        self.step_limit = 20
        
        self.total_reward = 0
        self.reward = 0
        
        # state
        self.state = np.zeros(shape=(8,))
        self.end = False
        
        # memory
        self.replay_memory = [] # [state(8), action(1), done(1), reward(1), next_state(8)] 
        
        # location and target
        self.location = environment.start_point
        self.target = environment.target_point      
    
    def action_and_next_state(self, show_q = False):
        
        if self.end:
            self.end = False
        
        self.state = np.copy(self.environment.get_state(self.location))
        
        
        Q_sa = self.estimator.predict(np.array([self.state]))
        
        memory_temp = np.zeros(shape=(19,)) # [state(8), action(1), done(1), reward(1), next_state(8)]
        
        ############## To do ##############
        memory_temp[:8] = np.copy(self.state) # Current state
        ############## To do ##############

        rand_factor = np.random.uniform(0, 1)

        if rand_factor < self.epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q_sa)
        if action == 0:
            _action = 'left'
            
        elif action == 1:
            _action = 'right'
            
        elif action == 2:
            _action = 'up'
            
        elif action == 3:
            _action = 'down'
        
        if show_q:
            print(Q_sa, _action)
        
        _, reward, done, next_state = np.copy(self.environment.get_reward_and_next_state(self.location, _action))
        
        self.location = np.int32(next_state[:2])
        
        self.reward = reward
        
        ############## To do ##############
        memory_temp[8] = np.copy(action) # action
        memory_temp[9] = np.float32(done) # done or not
        memory_temp[10] = np.copy(reward) # reward
        memory_temp[11:] = np.copy(next_state) # next state
        ############## To do #############

        prob = sigmoid(self.reward*10) # probabilistic sampling
        rand_factor = np.random.uniform() # probabilistic sampling
        
                
        
        """
        ############## Push current memory to replay memory ##############
        """ 
        if(rand_factor < prob):
            
            if(len(self.replay_memory) >= self.memory_size):
                ############## To do ##############
                self.replay_memory[np.random.randint(len(self.replay_memory))] = np.copy(memory_temp)
            
            else:
                self.replay_memory.append(np.copy(memory_temp))
                ############## To do ##############
                
        
        self.step += 1
        
        if reward == 1 or self.step > self.step_limit:
            
            """
            Reset position
            """
            self.end = True
            self.location = np.copy(self.environment.start_point)
            self.step = 0
         
    def train_estimator(self):
        nb_train = np.int32(self.memory_size*self.train_ratio)
        rand_arr = np.arange(nb_train)
        np.random.shuffle(rand_arr)
        
        replay_memory = np.zeros(shape=(nb_train, 19))
        replay_memory = np.copy(self.replay_memory)[np.int32(rand_arr)]
        
        state = np.copy(replay_memory[:, :8])
        
        ############## To do ##############
        next_state = np.copy(replay_memory[:, 11:])       
        Q_sa_next = np.copy(self.estimator.predict(next_state))
        ############## To do ##############

        
        """
        Make target q value
        """
        Q_sa = np.copy(self.estimator.predict(state))

        for i in range(nb_train):
            
            ############## To do ##############

            if replay_memory[i, 10] == 1:
                Q_sa[i][np.int32(replay_memory[i, 8])] = np.copy(replay_memory[i, 10])
            else:
                Q_sa[i][np.int32(replay_memory[i, 8])] = np.copy(replay_memory[i, 10] + self.discount_reward*np.max(Q_sa_next[i,:]))
            
            ############## To do ##############

        x_train = replay_memory[:, :8]
        target_q = np.copy(Q_sa)     

        for i in range(1):
            self.estimator.fit(x=x_train, y=target_q, epochs=1, verbose=None)


np.random.seed(35)
grid_size = 4
simple_maze = Simple_Maze(grid_size, [1,1], [grid_size,grid_size+1], 0.2)
agent = Agent(simple_maze)
img = simple_maze.show(agent.location)
plt.imshow(img, 'gray')

print(simple_maze.grid, 'gray')


## train #########################################

np.random.seed(1)

nb_episode = 370
cnt = 0

show_step = 10
reward = 0

while cnt < nb_episode:
    
    agent.action_and_next_state()
    
    reward += agent.reward
    
    if(agent.end == True):
        if cnt % show_step == 0:
            print("Episod : "+str(cnt) +"  Total reward : "+str(np.round(reward/show_step,2)) 
                  + "  Epsinon : "+str(np.round(agent.epsilon, 3)) + "  NB Memory : "+str(len(agent.replay_memory)))
            reward = 0
        cnt += 1
        
        
    if(len(agent.replay_memory) >= agent.memory_size):
            
        error = agent.train_estimator()
        if(agent.epsilon > agent.epsilon_min):
            agent.epsilon *= agent.discount_epsilon



## check result #########################################
images = []
agent.epsilon = 0
for i in range(100):
    
    agent.action_and_next_state()
    img = agent.environment.show(agent.location)
    images.append(img)
    
images = np.array(images)


### %matplotlib inline

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from time import sleep

n = images.shape[0]
a = np.zeros((n, n))
plt.figure()

for i in range(n):
    plt.imshow(images[i], 'gray')
    print
    plt.show()
    a[i, i] = 1
    sleep(0.1)
    clear_output(wait=True)

