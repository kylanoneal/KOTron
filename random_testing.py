from BMTron import *

if __name__ == '__main__':
    decay = 0.99
    curr_decay = 1
    for i in range(500):
        curr_decay *= decay
        print(curr_decay)