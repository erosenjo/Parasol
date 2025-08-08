from copy import deepcopy
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class DataReporter:
    def __init__(self):
        self.frames = []
        self.episode = 0

    # state is actually just the last three entries in the state vector
    def add_frame(self, reward, background, state):
        self.frames.append({
            "reward": reward,
            "background": background,
            "state": deepcopy(state),  # average error can be found in state
        })

    def report(self):
        # FIXME figure out what metrics to display
        # print("Data from steps:")
        # print(self.frames)
        errs = [x["state"][-1] for x in self.frames]
        actions = [int(x["state"][-2]) for x in self.frames]
        avg_err = sum(errs) / len(errs)
        eprint("Overall average error: {}".format(avg_err))
        eprint("Actions that led to this error:\n{}".format(actions))

    def start_or_advance_step(self, episodes):
        if self.episode < 1:
            eprint()
        self.episode += 1
        eprint(
            f"\rTraining: episode {self.episode}/{episodes}",
            end='', flush=True
        )
        if self.episode == episodes:
            eprint()
