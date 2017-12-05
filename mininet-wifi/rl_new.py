#!/usr/bin/python

"""
Setting mechanism to optimize the use of the APs
"""
import re
import random
import time
import datetime
import numpy as np
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelAP
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel
#from RL_brain import DeepQNetwork
#from display import Display
from mininet.wifiMobility import mobility
from mininet.wifiLink import Association
from mininet.wifiAssociationControl import associationControl
from mininet.log import  error, debug, setLogLevel
from mininet.util import   waitListening

class object():
    def __init__(self):
        self.set()
        self.topology()

    def chanFunt(self, new_ap, new_st):

        """collect rssi from aps to station
           :param new_ap: access point
           :param new_st: station
        """
        APS = ['ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7', 'ap8', 'ap9']
        for number in APS:
            if number == str(new_ap):
                indent = 0
                for item in new_st.params['apsInRange']:
                    if number in str(item):
                        for each_item in new_ap.params['stationsInRange']:
                            if str(new_st) in str(each_item):
                                return new_ap.params['stationsInRange'][each_item]
                                # a.append(ap1.params['stationsInRange'].values()[level+1])
                            else:
                                pass
                    else:
                        indent = indent + 1
                if indent == len(new_st.params['apsInRange']):
                    return 0
            else:
                pass

    def setSNR(self, signal):
        """
        set SNR
        :param signal: RSSI
        """
        if signal != 0:
            snr = float('%.2f' % (signal - (-91.0)))
        else:
            snr = 0
        return snr


    def connected_tag(self, new_string):
        """output the number of associated AP"""
        for each_item in new_string.params['associatedTo']:
            if str('ap1') in str(each_item):
                return 1
            elif str('ap2') in str(each_item):
                return 2
            elif str('ap3') in str(each_item):
                return 3
            elif str('ap4') in str(each_item):
                return 4
            elif str('ap5') in str(each_item):
                return 5
            elif str('ap6') in str(each_item):
                return 6
            elif str('ap7') in str(each_item):
                return 7
            elif str('ap8') in str(each_item):
                return 8
            elif str('ap9') in str(each_item):
                return 9
            else:
                return 0

    def loadness(self, new_ap):

        """ return the loadness of the APs"""
        return len(new_ap.params['associatedStations'])

    def sleeptime(self, hour, minu, sec):

        """time to output"""
        return hour * 3600 + minu * 60 + sec

    def step(self, currentID, action):
        actionID = action.argmax()
        if actionID != currentID and actionID <= n_APs:
            self.handover()
        reward = self._getreward(self.sta14, self.h1)
        nextstate = [self.chanFunt(self.ap1, self.sta14),self. chanFunt(self.ap2, self.sta14), self.chanFunt(self.ap3, self.sta14)]

        return reward, nextstate

    def handover(self, sta, ap, wlan):
        '''
        the function to realize transfer association
        :param sta: station
        :param ap: target_AP
        :param wlan:  OVSAP
        :return:  new connection
        '''
        changeAP = True
        """Association Control: mechanisms that optimize the use of the APs"""
        if sta.params['associatedTo'][wlan] == '' or changeAP == True:
            if ap not in sta.params['associatedTo']:
                cls = Association
                # debug('iwconfig %s essid %s ap %s\n' % (sta.params['wlan'][wlan], ap.params['ssid'][0], \
                #                                           ap.params['mac'][0]))
                # sta.pexec('iwconfig %s essid %s ap %s' % (sta.params['wlan'][wlan], ap.params['ssid'][0], \
                #                                           ap.params['mac'][0])
                cls.associate_noEncrypt(sta, ap, wlan)
                mobility.updateAssociation(sta, ap, wlan)

    def _parseIperf(self, iperfOutput):

        # r = r'([\d\.]+ \w+/sec)'
        r = r'([\d\.]+\w)'
        m = re.findall(r, iperfOutput)
        if m:
            return m[-1]
        else:
            # was: raise Exception(...)
            error('could not parse iperf output: ' + iperfOutput)
            return ''

    def iperf(self, hosts=None, l4Type='TCP', udpBw='10M',
              seconds=5, port=5001):
        hosts = hosts or [hosts[0], hosts[-1]]
        assert len(hosts) == 2
        client, server = hosts
        conn1 = 0
        conn2 = 0
        if client.type == 'station' or server.type == 'station':
            if client.type == 'station':
                while conn1 == 0:
                    conn1 = int(client.cmd('iwconfig %s-wlan0 | grep -ic \'Link Quality\'' % client))
            if server.type == 'station':
                while conn2 == 0:
                    conn2 = int(server.cmd('iwconfig %s-wlan0 | grep -ic \'Link Quality\'' % server))
                    # output( '*** Iperf: testing', l4Type, 'bandwidth between',
                    #        client, 'and', server, '\n' )
                    #        if not conn2:
                    #            print "#$%^%$#@#$%^%$#@"
                    #        else:
        server.cmd('killall -9 iperf')
        iperfArgs = 'iperf -p %d ' % port
        bwArgs = ''
        if l4Type == 'UDP':
            iperfArgs += '-u '
            bwArgs = '-b ' + udpBw + ' '
            # elif l4Type != 'TCP':
            #    raise Exception( 'Unexpected l4 type: %s' % l4Type )
            # if fmt:
            #   iperfArgs += '-f %s ' % fmt
        server.sendCmd(iperfArgs + '-s')
        if l4Type == 'TCP':
            if not waitListening(client, server.IP(), port):
                raise Exception('Could not connect to iperf on port %d'
                                % port)
        cliout = client.cmd(iperfArgs + '-t %d -c ' % seconds +
                            server.IP() + ' ' + bwArgs)
        # print cliout
        debug('Client output: %s\n' % cliout)
        servout = ''
        count = 2 if l4Type == 'TCP' else 1

        while len(re.findall('/sec', servout)) < count:
            servout += server.monitor(timeoutms=5000)

        server.sendInt()
        servout += server.waitOutput()
        debug('Server output: %s\n' % servout)
        result = [self._parseIperf(servout), self._parseIperf(cliout)]
        # if l4Type == 'UDP':
        #    result.insert( 0, udpBw )
        # output( '%s\n' % result )
        # return result[-1]
        return result[0]

    def _getreward(self, station, host):
        reward_dic = []
        reward_dic.append(self.iperf[station, host], l4Type='TCP', second=0.00001, port=5001)
        if len(reward_dic) == 3:
            return reward_dic[-1]
            # sum(reward_dic)/len(reward_dict)
    def set(self):
        setLogLevel('info')

    def topology(self):
            "Create a network."
            net = Mininet(controller=Controller, link=TCLink, accessPoint=OVSKernelAP)

            print "*** Creating nodes"
            self.sta14 = net.addStation('sta14', mac='00:00:00:00:00:15', ip='10.0.0.15/8', range=50)
            self.h1 = net.addHost('h1', mac='00:00:00:00:00:01', ip='10.0.0.1/8')
            self.ap1 = net.addAccessPoint('ap1', ssid='ssid-ap1', mode='g', channel='1', position='200,30,0', range=180)
            self.ap2 = net.addAccessPoint('ap2', ssid='ssid-ap2', mode='g', channel='1', position='100,230,0', range=180)
            self.ap3 = net.addAccessPoint('ap3', ssid='ssid-ap3', mode='g', channel='1', position='300,230,0', range=180)

            c1 = net.addController('c1', controller=Controller)

            print "*** Configuring wifi nodes"
            net.configureWifiNodes()

            print "*** Associating and Creating links"
            net.addLink(self.h1, self.ap1)
            net.addLink(self.ap1,self. ap2)
            net.addLink(self.ap2, self.ap3)

            """uncomment to plot graph"""
            net.plotGraph(max_x=400, max_y=400)

            """association control"""
            net.associationControl("ssf")

            """Seed"""
            net.seed(5)
            """random walking"""
            net.startMobility(time=0, model='RandomDirection', max_x=400, max_y=400, min_v=3, max_v=5)

            print "*** Starting network"
            net.build()
            c1.start()
            self.ap1.start([c1])
            self.ap2.start([c1])
            self.ap3.start([c1])

            print "*** Running CLI"
            CLI(net)
            second = self.sleeptime(0, 0, 1)

            # new_rssi = [chanFunt(ap1,sta14),chanFunt(ap2,sta14),chanFunt(ap3,sta14)]
            # n_actions, n_APs = len(new_rssi), len(new_rssi)
            # brain = DeepQNetwork(n_actions,n_APs,param_file = None)

            # state = new_rssi
            # print 'initial observation:' + str(state)
            try:
                while True:
                    time.sleep(second)
                    new_rssi = [self.chanFunt(self.ap1, self.sta14), self. chanFunt(self.ap2, self.sta14), self. chanFunt(self.ap3, self.sta14)]
                    print new_rssi,  self.connected_tag(self.sta14)
                    # print _getreward(sta14,h1)
                    print self.iperf([self.sta14, self.h1], seconds=0.0000001)
                    # action,q_value = brain.choose_action(state)
                    # reward, nextstate = step(rssi_tag(sta14),action)

                    # brain.setPerception(state, action, reward, nextstate)
                    # state = nextstate
            except KeyboardInterrupt:
                print 'saving replayMemory...'
                # brain.saveReplayMemory()
            pass
            # print new_rssi
            # snr_dict = map(setSNR,new_rssi)



            print "*** Stopping network"
            net.stop()
