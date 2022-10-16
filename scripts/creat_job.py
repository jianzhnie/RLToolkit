import subprocess
import time

ALGOS = ['c51', 'sac', 'dqn', 'fqf', 'ppo', 'qrdqn', 'rainbow', 'sac']
ENVS = [
    'PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'EnduroNoFrameskip-v4',
    'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SeaquestNoFrameskip-v4',
    'SpaceInvadersNoFrameskip-v4'
]

for algo in ALGOS:
    script = 'atari_' + algo + '.py'
    for env_id in ENVS:
        command = ['python3', script, '--task', env_id, '--logger', 'wandb']
        ok = subprocess.run(command)
        time.sleep(10)
