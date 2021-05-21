import torch
import numpy as np
from torch.utils.data import Dataset
from GameGen import Game
import random

class gameDataset(Dataset):
    def __init__(self, board_shape, time, size):
        self.data = []
        for i in range(size):
            p_alive = random.randint(30,60)
            g = Game.from_rand(board_shape, p_alive/100)
            g.run_to(time)
            board_size = board_shape[0]*board_shape[1]
            initial_board, final_board = torch.flatten(torch.from_numpy(g.board[0])).float(), torch.flatten(torch.from_numpy(g.board[-1])).float()
            initial_board += torch.where(initial_board < 0.1, torch.rand(board_size)/4, -1*torch.rand(board_size)/4)
            self.data.append((initial_board, final_board))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"initial": self.data[idx][0].float(), "final":self.data[idx][1].float()}

class equalityDataset(Dataset):
    def __init__(self, board_shape, size):
        self.data = []
        for i in range(size):
            board_size = board_shape[0]*board_shape[1]
            board = np.random.randint(0, 2, size=board_shape).astype(float)
            board2 = np.copy(board)
            makeEqual = random.randint(0,1)
            if makeEqual == 0:
                flip_only_one = random.randint(0,1)
                if flip_only_one == 0:
                    board2 = np.random.randint(0, 2, size=board_shape).astype(float)
                else:
                    index1, index2 = random.randint(0,board_shape[0]-1), random.randint(0,board_shape[0]-1)
                    board2[index1, index2] = 1 - board2[index1, index2]
            board += np.where(board < 0.1, np.random.rand(*board_shape)/4, -1*np.random.rand(*board_shape)/4)
            
            self.data.append((torch.flatten(torch.from_numpy(board)), torch.flatten(torch.from_numpy(board2)), makeEqual))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"game1": self.data[idx][0].float(), "game2": self.data[idx][1].float(), "label": torch.Tensor([self.data[idx][2]]).float()}
