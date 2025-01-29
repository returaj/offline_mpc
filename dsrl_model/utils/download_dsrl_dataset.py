import sys
import gymnasium as gym
import dsrl.offline_safety_gymnasium  # type: ignore


def main(task):
    eval_env = gym.make(task)
    _ = eval_env.get_dataset()


if __name__ == "__main__":
    task = sys.argv[1]
    main(task)
