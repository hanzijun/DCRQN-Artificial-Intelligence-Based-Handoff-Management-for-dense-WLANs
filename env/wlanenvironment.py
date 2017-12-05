''''
This module is  the SWAN environment:
 observation : the RSSI values at two AP (192.168.0.124 and 192.168.0.131)
 action :             handover to the APs, e.g. array[1,0] stands for handoff to AP1
 reward:            wireless  throughput
'''''

import threading
import urllib2
from time import sleep
import numpy as np
import json
import time
import math

class wlanEnv:
    def __init__(self, remoteControllerAddr, seqLen, timeInterval=0.1, no_guarantee=False):
        self.remoteAddr = remoteControllerAddr
        self.numAp = 0
        self.seqLen = seqLen
        self.timeInverval = timeInterval
        self.threads = []
        self.end = False
        self.no_guarantee = no_guarantee

        self.macAddr = '58:fb:84:90:a0:b0'    #''88:cb:87:67:6b:29'
        rssiUrl = 'http://' + self.remoteAddr + "/dqn/rssi/json?mac=" + self.macAddr
        rssiDict = curl_keystone(rssiUrl)
        rssiDict = json.loads(rssiDict)
        dictKey = rssiDict.keys()
        dictKey.remove('state')
        self.numAp = len(dictKey)
        self.ap2id = dict(zip(dictKey, xrange(0, self.numAp)))
        self.id2ap = dict(zip(xrange(0, self.numAp), dictKey))
        self.obsevation = None
        self.reward = None
        self.valid = False

        # initial actionId, currentId
        self.lastActionId = self.numAp
        self.currentId = self.__getCurrentId()

    def __getCurrentId(self):
        """
        return the current serving AP's number
        :return:
        """
        url = 'http://' + self.remoteAddr + '/odin/clients/connected/json'
        dict = curl_keystone(url)
        dict = json.loads(dict)
        print dict[self.macAddr]
        agentIp = dict[self.macAddr]['agent']
        agentId = self.ap2id[agentIp]
        return agentId


    def __getStatesFromRemote(self, clientHwAddr, timeInterval):
        '''
        refresh the environment to reach the observation
        '''
        while not self.end:
            try:
                rssiUrl = 'http://' + self.remoteAddr + '/dqn/rssi/json?mac=' + clientHwAddr
                rssiDict = curl_keystone(rssiUrl)
                rssiDict = json.loads(rssiDict)
                rewardUrl = 'http://' + self.remoteAddr + '/dqn/reward/json?mac=' + clientHwAddr
                rewardDict = curl_keystone(rewardUrl)
                rewardDict = json.loads(rewardDict)
            except:
                print 'Error or Exception in __getStatesFromRemote()'
            else:
                if len(rssiDict) == (self.numAp + 1) and len(rewardDict) == 2:
                    if rssiDict['state'] and rewardDict['state']:
                        rssiDict.pop('state')
                        rewardDict.pop('state')

                        if self.obsevation is None :
                            self.obsevation = np.array([rssiDict.values()])
                        elif self.obsevation.shape[0] == self.seqLen:
                            obsevation = np.delete(self.obsevation, (0), axis=0)
                            obsevation = np.append(obsevation, [rssiDict.values()],axis=0)
                            self.obsevation = obsevation
                            if not self.valid:
                                self.valid = True
                        else:
                            self.obsevation = np.append(self.obsevation, [rssiDict.values()], axis=0)

                        if self.reward is None:
                            self.reward = np.array([rewardDict['reward']])
                        elif self.reward.shape[0] == (self.seqLen//2):
                            reward = np.delete(self.reward, (0), axis=0)
                            reward = np.append(reward, [rewardDict['reward']], axis=0)
                            self.reward = reward
                        else:
                            self.reward = np.append(self.reward, [rewardDict['reward']], axis=0)
                else:
                    print "Some ap is not working......Please check!!!"
            finally:
                sleep(timeInterval)

    def __handover(self, clientHwAddr, agentIp):
        """
        handover to 192.168.0.124/192.168.0.131
        :param clientHwAddr:
        :param agentIp:
        :return:
        """
        handoverUrl = 'http://' + self.remoteAddr + '/dqn/handover/json?mac=' + clientHwAddr + '&&agent=' + agentIp
        print "handover to %s"%(agentIp)
        curl_keystone(handoverUrl)

    '''
    @:returns
    input vector dimension
    action space dimension
    '''

    def getDimSpace(self):
        return self.numAp, self.numAp              # numActions

    def observe(self):
        # print self.obsevation
        rssi = self.obsevation.astype(int)
        return self.valid, rssi

    def step(self, action):
        """
        reinforcement learning important step:
        do actions and arrive the next state ,return the reward of action and next state
        :param action: array[1,0]
        :return:
        """
        actionId = action.argmax()
        if  actionId != self.currentId:
            self.__handover(self.macAddr, self.id2ap[actionId])
            if not self.no_guarantee:
                sleep(self.timeInverval * self.seqLen)
            self.currentId = actionId

        _, reward, throught = self.getReward()
        self.lastActionId = actionId
        _, nextObservation = self.observe()
        nextstate = sum(nextObservation)/len(nextObservation)

        return reward, throught, nextstate

    def getReward(self):
        self.throught = self.reward.mean()
        # print self.throught, self.reward.std()
        return self.valid, self.throught, self.throught

    def start(self):
        t1 = threading.Thread(target=self.__getStatesFromRemote, args=(self.macAddr, self.timeInverval))
        self.threads.append(t1)
        for t in self.threads:
            t.setDaemon(True)
            t.start()
        print 'start'

    def stop(self):
        self.end = True
        for t in self.threads:
            t.join()
        print 'stop'

def curl_keystone(url):
    req = urllib2.Request(url)
    response = urllib2.urlopen(req)
    return response.read()

if __name__ == '__main__':
    env = wlanEnv('10.103.238.19:8080', 10, timeInterval=0.1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    # sleep(1)
    # print env.cal()
    env.start()
    print env.ap2id
    print env.id2ap
    sleep(1)
    print env.step(np.array([1,0]))

    sleep(1)
    print env.step(np.array([0,1]))

    sleep(1)
    print env.step(np.array([0,1]))

    sleep(1)
    print env.step(np.array([1,0]))

    sleep(1)
    print env.step(np.array([1, 0]))
    sleep(1)
    # print env.step(np.array([0,0,1]))
    # sleep(1)
    # print env.step(np.array([0,0,1]))
    # sleep(1)
    # print env.step(np.array([1,0,0]))
    # sleep(1)
    # print env.step(np.array([0,0,1]))
    # sleep(1)
    # print env.step(np.array([0,0,1]))
    sleep(1)
    env.stop()
    sleep(2)
    pass
