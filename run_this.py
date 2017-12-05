from env.wlanenvironment import wlanEnv
from RL_brain import DeepQNetwork
import time
import numpy as np
from display import Display
# import signal

CONTROLLER_IP = '10.103.238.19:8080'
BUFFER_LEN = 10
ENV_REFRESH_INTERVAL = 0.1


def train():

    env = wlanEnv(CONTROLLER_IP, BUFFER_LEN, timeInterval=ENV_REFRESH_INTERVAL)
    env.start()
    n_actions, n_APs = env.getDimSpace()
    brain = DeepQNetwork(n_actions, n_APs, param_file=None)        #fixme: param_file need to be restored?
    while not env.observe()[0]:
        time.sleep(0.5)

    observation = env.observe()[1]
    state = sum(observation)/len(observation)
    np.set_printoptions(threshold=5)
    print 'Initial observation:\n' + str(state)

    data = {}
    fig = Display(env.id2ap)
    fig.display()
    try:
        while True:
            action, q = brain.choose_action(state)
            print 'action:\n' + str(action.argmax())
            reward, throught, nextstate = env.step(action)
            print 'reward: ' + str(reward) + ', throught: ' + str(throught)
            print 'Next state:\n' + str(nextstate)

            data['timestamp'] = time.time()
            data['rssi'] = nextstate[-1]
            data['q'] = q
            data['reward'] = reward
            data['action_index'] = np.argmax(action)
            fig.append(data)

            brain.setPerception(state, action, reward,nextstate)
            state = nextstate
    except KeyboardInterrupt:
        print 'Saving replayMemory......'
        brain.saveReplayMemory()
        fig.stop()
    pass



if __name__ == "__main__":
    train()
