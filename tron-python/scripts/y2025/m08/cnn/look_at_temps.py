from tron.ai import MCTS

visits = [[400, 300, 300], [700, 300], [700, 200, 100], [850, 100, 50]]

ts = [1.0, 0.5]

for v in visits:
    for t in ts:

        MCTS.softmax_sample(v, t)