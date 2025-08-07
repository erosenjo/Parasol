from copy import deepcopy


class DataReporter:
    def __init__(self):
        self.frames = []

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
        print("Overall average error: {}".format(avg_err))
        print("Actions that led to this error:\n{}".format(actions))
