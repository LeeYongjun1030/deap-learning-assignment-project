# 2020 12 05 Seungwon Choi
# csw3575@snu.ac.kr

import numpy as np

class Simple_Maze:
    def __init__(self, grid_size, start_point, target_point, wall_ratio):
        
        self.start_point = start_point
        self.target_point = target_point
        
        self.grid_size = grid_size
        self.wall_ratio = wall_ratio
        self.nb_grid_cell = np.int32(self.grid_size*self.grid_size)
        self.nb_trap_cell = np.int32(self.nb_grid_cell*self.wall_ratio)
               
        self.grid = np.zeros(shape=(self.nb_grid_cell,))
        self.grid_aux = np.zeros(shape=(self.grid_size+2, self.grid_size+2))
        
        
        rand_arr = np.arange(self.nb_grid_cell)
        np.random.shuffle(rand_arr)
        
        if self.nb_trap_cell > 0:
            self.grid[rand_arr[:self.nb_trap_cell]] = -1.0        
        
        self.grid = np.reshape(self.grid, (grid_size, grid_size))
        
        self.grid[start_point[0], start_point[1]] = 0.0
        
        self.grid_aux[:, 0] = -1.0
        self.grid_aux[:, self.grid_size+1] = -1.0
        self.grid_aux[0, :] = -1.0
        self.grid_aux[self.grid_size+1, :] = -1.0
        self.grid_aux[1:self.grid_size+1, 1:self.grid_size+1] = np.copy(self.grid)
        self.grid = np.copy(self.grid_aux)
        
        
        self.grid[target_point[0], target_point[1]] = 1.0
        
    def show(self, location):
        
        wall = np.copy(self.grid == -1)
        route = np.copy(self.grid == 0)
        target = np.copy(self.grid == 1)
        
        image = np.zeros(shape=(self.grid_size+2, self.grid_size+2))
        image[wall] = 0.0
        image[route] = 1.0
        image[target] = 1.0
        image[location[0], location[1]] = 0.5
        
        return image
    
    def get_state(self, location):
        row, col = location[0], location[1]
        
        left, right, up, down = self.grid[row, col-1], self.grid[row, col+1], self.grid[row-1, col], self.grid[row+1, col]
        
        return np.array([row, col, left, right, up, down, self.target_point[0], self.target_point[1]])
        
        
        
    def get_reward_and_next_state(self, location, action, end=False):
        
        reward = -0.01
        
        location = np.copy(np.int32(location))
        
        row, col = location[0], location[1]
        
        # action
        if action == 'left' and self.grid[row, col-1] != -1:
            _action = 0
            location[1] -= 1
            
        elif action == 'right' and self.grid[row, col+1] != -1:
            _action = 1
            location[1] += 1
        
        elif action == 'up' and self.grid[row-1, col] != -1:
            _action = 2
            location[0] -= 1
            
        elif action == 'down' and self.grid[row+1, col] != -1:
            _action = 3
            location[0] += 1
        
        row, col = location[0], location[1]
        #print(self.grid[row, col])
        #print('here')
        
        # reward
        if (self.grid[row, col] == 1):
        #    print("Ya")
            reward = 1
        
        done = False
        # next state
        if (end):
            location = np.copy(np.int32(self.start_point))
            done = True
            
        if (reward == 1):
            location = np.copy(np.int32(self.start_point))
            done = True
            
        row, col = location[0], location[1]
        #print(row, col, done, reward, self.grid[row, col])
            
        left, right, up, down = self.grid[row, col-1], self.grid[row, col+1], self.grid[row-1, col], self.grid[row+1, col]
        
        return action, reward, done, np.array([row, col, left, right, up, down, self.target_point[0], self.target_point[1]])
