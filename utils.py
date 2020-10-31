import sys, os
from os.path import join
import matplotlib.pyplot as plt


CUR = os.path.abspath(os.path.dirname(__file__))

def bottom_n_percent(l, n):
    n = int(len(l) * n / 100.)
    if n == 0:
        n = 1
    return sum(sorted(l)[:n]) / n

def plot_metrics(metrics_file):
    with open(metrics_file, 'r') as file_:
        lines = file_.readlines()[6:]
    mu, bot, loss = [], [], []
    for line in lines:
        line = line.rstrip().split(',')
        mu.append(float(line[0].split(':')[-1]))
        bot.append(float(line[2].split(':')[-1]))
        loss.append(float(line[3].split(':')[-1]))
    x = list(range(len(mu)))

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(x, loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Steps/ep', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, mu, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:green'
    # ax3.set_ylabel('BotSteps/ep', color=color)  # we already handled the x-label with ax1
    # ax3.plot(x, bot, color=color)
    # ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

if __name__ == '__main__':
    metrics_path = join(CUR, 'dueling_ddqn_trained.txt')
    metrics_path = join(CUR, 'big_dueling_ddqn_trained.txt')
    plot_metrics(metrics_path)