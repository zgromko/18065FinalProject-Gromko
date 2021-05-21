import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

neighbor_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])

class Game:
    def __init__(self, board: np.array):
        self.shape = board.shape
        self.board = [board]
        self.step = 0

    @staticmethod
    def from_rand(shape: tuple, p_alive):
        board = np.random.randint(0, 100, size=shape)
        board = (board > (100-100*p_alive)).astype(int)
        return Game(board)

    def neighbors_conv(self):
        return convolve(self.board[-1], neighbor_kernel, mode='constant', cval = 0)

    def next_state(self):
        neighbors = self.neighbors_conv()
        self.board.append(self.board[-1] & (neighbors == 2))
        self.board[-1] |= (neighbors == 3)
        self.step += 1

    def run_to(self, time):
        for t in range(time):
            self.next_state()

    def animate(self, time_steps, rate):
        plt.ion()
        self.im = plt.imshow(self.board[0], vmin=0,vmax=2,cmap=plt.cm.gray)
        i = 0
        while i < time_steps:
            self.im.set_data(self.board[i])
            i += 1
            if i > self.step:
                self.next_state()
            plt.pause(1/rate)

    def show(self, step):
        plt.ion()
        self.im = plt.imshow(self.board[step],vmin=0,vmax=2,cmap=plt.cm.gray)
