# from multiprocessing import Pool

# def f(x):
#     import controlRB
#     lbm = controlRB.lbmd2q9(mode='cuda',device=x,verbose=True,NX=20,NY=20,Ra=1e4,name=str(x))

# if __name__ == '__main__':
#     with Pool(2) as p:
#         print(p.map(f, [1,3]))
import gym
import gym_RB
import numpy as np

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var().unwrapped
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'reset_group':
                observation = env.reset_group(data)
                remote.send(observation)
            elif cmd == 'render':
                raise NotImplementedError
                #remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'num_envs':
                remote.send(env.num_envs)
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break

from multiprocessing import Process, Pipe
import cloudpickle
class CloudpickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)

from stable_baselines.common.policies import MlpPolicy,CnnPolicy
from stable_baselines import PPO2


import os
setGPUs = [1,3]
if isinstance(setGPUs,int):
    setGPUs = [setGPUs]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(s) for s in setGPUs])

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85 # use a reasonable fraction of the memory
# tf.keras.backend.set_session(tf.Session(config=config))


num_process = 2
num_env_per_process = 16
env_id = 'RB-vec-v0'
env_kwargs = dict(cuda=True,NX=20,Ra=1e4,dt=20,actionN=10,discrateN=2,probed=(4,4),verbose=2)

process_kwargs = [dict(device=0,name=0),dict(device=1,name=1)] 
env_kwargs_per_process = [{**env_kwargs,**p} for p in process_kwargs]

def getf(env_id,num_env_per_process,env_kwargs_per_process_now,i):
    return lambda: gym.make(env_id,num_envs=num_env_per_process,**env_kwargs_per_process_now[i])
env_fns = [getf(env_id,num_env_per_process,env_kwargs_per_process,i) for i in range(num_process)]


remotes, work_remotes = zip(*[Pipe() for _ in range(num_process)])
processes = [Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                  for (work_remote, remote, env_fn) in zip(work_remotes, remotes, env_fns)]

for process in processes:
    process.daemon = True  # if the main process crashes, we should not cause things to hang
    process.start()
for remote in work_remotes:
    remote.close()


print('resetting env...')
for remote in remotes:
    remote.send(('reset', None))
res = np.concatenate([remote.recv() for remote in remotes],axis=0)
print('results:',res)
print('shape:',res.shape)


for remote in remotes:
    remote.send(('close', None))
for process in processes:
    process.join()

print('done')