#!/usr/bin/python

"""
Setting mechanism to optimize the use of the APs
"""
import threading
import time
import model
import re
import datetime
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelAP
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.link import Association
from mininet.wifiAssociationControl import associationControl
from mininet.wifiChannel import setChannelParams
from mininet.wifiMobility import mobility
from mininet.log import info, error, debug, output, warn,setLogLevel
from mininet.util import (quietRun, fixLimits, numCores, ensureRoot,
                           macColonHex, ipStr, ipParse, netParse, ipAdd,
                           waitListening)

def _timeregul(tcpout):
    r = r'\d{2}\:\d{2}\:\d{2}'
    m = re.findall(r,tcpout)
    if m:
        return  m[-1]
    else:
        print "***could not find time***"

def tcpexamine(host = None,port = 5001):
    #while 1==1:
        T = 'tcpdump tcp'
        tcpout = host.cmd(T + '-i %s-eth0 port %d'%(host,5001))
        result = _timeregul(tcpout)
        print result

def _parseIperf(iperfOutput):
 
    #r = r'([\d\.]+ \w+/sec)'
    r = r'([\d\.]+\w)'
    m = re.findall(r, iperfOutput)
    if m:
        return m[-1]
    else:
        # was: raise Exception(...)
        error('could not parse iperf output: ' + iperfOutput)
        return ''

def iperf( hosts=None, l4Type='TCP',
           seconds=5, port=5001): 
    hosts = hosts or [ hosts[ 0 ], hosts[ -1 ] ]
    assert len( hosts ) == 2
    client, server = hosts
    if not client.params['associatedTo'] :
        result=['0']
    else:
        conn1 = 0
        conn2 = 0
        if client.type == 'station' or server.type == 'station':
            if client.type == 'station':
                while conn1 == 0:
                    conn1 = int(client.cmd('iwconfig %s-wlan0 | grep -ic \'Link Quality\'' % client))
            if server.type == 'station':
                while conn2 == 0:
                    conn2 = int(server.cmd('iwconfig %s-wlan0 | grep -ic \'Link Quality\'' % server))
    #output( '*** Iperf: testing', l4Type, 'bandwidth between',
    #        client, 'and', server, '\n' )
        server.cmd( 'killall -9 iperf' )
        iperfArgs = 'iperf -p %d ' % port
        bwArgs = ''
        #if l4Type == 'UDP':
        #    iperfArgs += '-u '
        #    bwArgs = '-b ' + udpBw + ' '
        #elif l4Type != 'TCP':
        #    raise Exception( 'Unexpected l4 type: %s' % l4Type )
        #if fmt:
        #    iperfArgs += '-f %s ' % fmt
        server.sendCmd( iperfArgs + '-s' )
        if l4Type == 'TCP':
            if not waitListening( client, server.IP(), port ):
                raise Exception( 'Could not connect to iperf on port %d'
                                 % port )
        cliout = client.cmd( iperfArgs + '-t %d -c ' % seconds +
                             server.IP() + ' ' + bwArgs )
        debug( 'Client output: %s\n' % cliout )
        servout = ''
        count = 2 if l4Type == 'TCP' else 1
        while len( re.findall( '/sec', servout ) ) < count:
            servout += server.monitor( timeoutms=5000 )
            #print "#$%^%$#@$%^%$#@$%^%$#"
        server.sendInt()
        servout += server.waitOutput()
        debug( 'Server output: %s\n' % servout )
        result = [_parseIperf(servout),_parseIperf(cliout)]
        #if l4Type == 'UDP':
        #    result.insert( 0, udpBw )
           #output( '%s\n' % result )
        #print result[0]
        return result[0]

def scan(sta):
    print '***start scan'
    sta.cmd('iw dev sta14-wlan0 scan|grep ssid')
    global timer
    timer = threading.Timer(2, fun_timer)
    timer.start()

def associationcontrol(sta,ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9):
        """associationcontrol function to scan channel and change AP when 
        the rssi below -48 dBm"""
    
        AP={}
        if not sta.params['associatedTo']:
            RSSI = 0 
        else:
            RSSI=sta.params['rssi'][0]
            if RSSI>-47.0:
                pass
            else:
                #APS=['ap1','ap2','ap3','ap4','ap5','ap6','ap7','ap8','ap9']
                try:
                    #R= [chanFunt(ap1,sta),chanFunt(ap2,sta),chanFunt(ap3,sta),
                    #    chanFunt(ap4,sta),chanFunt(ap5,sta),chanFunt(ap6,sta),
                    #    chanFunt(ap7,sta),chanFunt(ap8,sta),chanFunt(ap9,sta)]
                    #if index%3 == 0:
                    print '***start scan'
                    #sta.cmd('iw dev sta14-wlan0 scan|grep ssid')
                    print ('1',datetime.datetime.now())
                    #sta.cmd('iw dev sta14-wlan0 scan freq "2462"')
                    sta.cmd('iw dev sta14-wlan0 scan trigger')
                    while True:
                        a = sta.cmd('iw dev sta14-wlan0 scan dump|grep ssid')
                        if len(a)==153:
                            break
                        else:
                            pass 
                    print (sta.cmd('iw dev sta14-wlan0 scan dump|grep ssid'),datetime.datetime.now())   
                    R = [chanFunt(ap1,sta),chanFunt(ap2,sta),chanFunt(ap3,sta),
                         chanFunt(ap4,sta),chanFunt(ap5,sta),chanFunt(ap6,sta),
                         chanFunt(ap7,sta),chanFunt(ap8,sta),chanFunt(ap9,sta)]
                    a = -100.0
                    for each_item in range(0,len(R)):
                        if R[each_item]!= 0:
                            if R[each_item]>a:
                                a = R[each_item]
                            else:
                                pass
                        else:
                            pass
                    N = R.index(a)
               # select the strongest signal from the AP                
                    if N == 0:
                        AP['1']=ap1
                    elif N==1:
                        AP['2']=ap2
                    elif N==2:
                        AP['3']=ap3
                    elif N==3:
                        AP['4']=ap4
                    elif N==4:
                        AP['5']=ap5
                    elif N==5:
                        AP['6']=ap6
                    elif N==6:
                        AP['7']=ap7
                    elif N==7:
                        AP['8']=ap8
                    elif N==8:
                        AP['9']=ap9
                    for idx, wlan in enumerate(sta.params['wlan']):
                       # if wlan == iface:
                        wlan = idx
                        break   
                    handover(sta,AP.values()[-1],wlan) 
                    print ('3',datetime.datetime.now())
                except:
                    pass

def handover(sta,ap,wlan):
        #debug('iw dev %s disconnect' % sta.params['wlan'][wlan])
        #sta.pexec('iw dev %s disconnect' % sta.params['wlan'][wlan])
        changeAP = True
        """Association Control: mechanisms that optimize the use of the APs"""
        if sta.params['associatedTo'][wlan] == '' or changeAP == True:
            if ap not in sta.params['associatedTo']:
                print "****start handover"
                cls = Association
                if 'encrypt' not in ap.params:
                    cls.associate_noEncrypt(sta, ap, wlan)
                else:
                    if ap.params['encrypt'][0] == 'wpa' or ap.params['encrypt'][0] == 'wpa2':
                        cls.associate_wpa(sta, ap, wlan)
                    elif ap.params['encrypt'][0] == 'wep':
                        cls.associate_wep(sta, ap, wlan)
                mobility.updateAssociation(sta, ap, wlan)


def sleeptime(hour,minu,sec):

    """time to output"""
    return hour*3600 + minu*60 + sec


def chanFunt (new_ap,new_st):

    """collect rssi from aps to station
       :param new_ap: access point
       :param new_st: station
    """
    APS=['ap1','ap2','ap3','ap4','ap5','ap6','ap7','ap8','ap9']
    for number in APS:
        if  number == str(new_ap):
            indent=0
            for item in new_st.params['apsInRange']:
                if number in str(item):
                    for each_item in new_ap.params['stationsInRange']:
                        if str(new_st) in str(each_item):
                            return new_ap.params['stationsInRange'][each_item]
                             #a.append(ap1.params['stationsInRange'].values()[level+1])
                        else:
                            pass
                else:
                    indent=indent+1
            if indent == len(new_st.params['apsInRange']):
                return 0
        else:
            pass


def loadness(new_ap):

    """ return the loadness of the APs"""
    return len(new_ap.params['associatedStations'])


def rssi_tag (new_string):

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

def average(List):
    """average level"""
    SUM = 0
    for i in range(0,len(List)):
        SUM = SUM + List[i]
    return SUM/len(List)

def topology():
    "Create a network."
    net = Mininet( controller=Controller, link=TCLink, accessPoint=OVSKernelAP )

    print "*** Creating nodes"
    #sta1 = net.addStation( 'sta1', mac='00:00:00:00:00:02', ip='10.0.0.2/8',position='50,60,0',range=15)
    #sta2 = net.addStation( 'sta2', mac='00:00:00:00:00:03', ip='10.0.0.3/8',position='65,70,0',range=15)
    #sta3 = net.addStation( 'sta3', mac='00:00:00:00:00:04', ip='10.0.0.4/8',position='50,55,0',range=15)
    #sta4 = net.addStation( 'sta4', mac='00:00:00:00:00:05', ip='10.0.0.5/8',position='110,65,0',range=15)
    #sta5 = net.addStation( 'sta5', mac='00:00:00:00:00:06', ip='10.0.0.6/8',position='15,90,0',range=15)
    #sta6 = net.addStation( 'sta6', mac='00:00:00:00:00:07', ip='10.0.0.7/8',position='10,100,0',range=15)
    #sta7 = net.addStation( 'sta7', mac='00:00:00:00:00:08', ip='10.0.0.8/8',position='68,20,0',range=15)
    #sta8 = net.addStation( 'sta8', mac='00:00:00:00:00:09', ip='10.0.0.9/8',position='55,100,0',range=15)
    #sta9 = net.addStation( 'sta9', mac='00:00:00:00:00:10', ip='10.0.0.10/8',position='95,25,0',range=15)
    #sta10 = net.addStation( 'sta10', mac='00:00:00:00:00:11', ip='10.0.0.11/8',position='110,20,0',range=15)
    #sta11 = net.addStation( 'sta11', mac='00:00:00:00:00:12', ip='10.0.0.12/8',position='110,10,0',range=15)
    #sta12 = net.addStation( 'sta12', mac='00:00:00:00:00:13', ip='10.0.0.13/8',position='20,110,0',range=15)
    #sta13 = net.addStation( 'sta13', mac='00:00:00:00:00:14', ip='10.0.0.14/8',position='110,60,0',range=15)
    #sta15 = net.addStation( 'sta15', mac='00:00:00:00:00:16', ip='10.0.0.16/8',position='110,100,0',range=15)
    #sta16 = net.addStation( 'sta16', mac='00:00:00:00:00:17', ip='10.0.0.17/8',position='100,110,0',range=15)
    #sta17 = net.addStation( 'sta17', mac='00:00:00:00:00:18', ip='10.0.0.18/8',position='95,105,0',range=15)
    #sta18 = net.addStation( 'sta18', mac='00:00:00:00:00:19', ip='10.0.0.19/8',position='105,105,0',range=15)
    #sta19 = net.addStation( 'sta19', mac='00:00:00:00:00:20', ip='10.0.0.20/8',position='68,105,0',range=15)
   #sta20 = net.addStation( 'sta20', mac='00:00:00:00:00:21', ip='10.0.0.21/8',range=15)   
    sta14 = net.addStation( 'sta14', mac='00:00:00:00:00:15', ip='10.0.0.15/8',range=15 )
    ap1 = net.addAccessPoint( 'ap1', ssid= 'ssid-ap1', mode= 'g', channel= '1', position='25,25,0', range=30 )
    ap2 = net.addAccessPoint( 'ap2', ssid= 'ssid-ap2', mode= 'g', channel= '6', position='25,60,0', range=30 )
    ap3 = net.addAccessPoint( 'ap3', ssid= 'ssid-ap3', mode= 'g', channel= '1', position='25,100,0', range=30 )
    ap4 = net.addAccessPoint( 'ap4', ssid= 'ssid-ap4', mode= 'g', channel= '11', position='60,25,0', range=30 )
    ap5 = net.addAccessPoint( 'ap5', ssid= 'ssid-ap5', mode= 'g', channel= '1', position='60,60,0', range=30 )
    ap6 = net.addAccessPoint( 'ap6', ssid= 'ssid-ap6', mode= 'g', channel= '11', position='60,100,0', range=30 )
    ap7 = net.addAccessPoint( 'ap7', ssid= 'ssid-ap7', mode= 'g', channel= '1', position='100,25,0', range=30 )
    ap8 = net.addAccessPoint( 'ap8', ssid= 'ssid-ap8', mode= 'g', channel= '6', position='95,60,0', range=30 )
    ap9 = net.addAccessPoint( 'ap9', ssid= 'ssid-ap9', mode= 'g', channel= '1', position='95,95,0', range=30 )
    h1 = net.addHost('h1', mac='00:00:00:00:00:01', ip='10.0.0.1/8')	
    #h2 = net.addHost('h2', mac='00:00:00:00:00:06', ip='10.0.0.6/8') 
    c1 = net.addController( 'c1', controller=Controller )

    print "*** Configuring wifi nodes"
    net.configureWifiNodes()

    print "*** Associating and Creating links"
    net.addLink(h1,ap1)
    net.addLink(ap1,ap2)
    net.addLink(ap2,ap3)
    net.addLink(ap3,ap6)
    net.addLink(ap6,ap5)
    net.addLink(ap5,ap4)
    net.addLink(ap4,ap7)
    net.addLink(ap7,ap8)
    net.addLink(ap8,ap9)
    #net.addLink(sta1,ap5)
    #net.addLink(sta2,ap5)
    #net.addLink(sta3,ap5)
    #net.addLink(sta4,ap8)
    #net.addLink(h2,ap2)
    print "*** Starting network"
    net.build()
    c1.start()
    ap1.start( [c1] )
    ap2.start( [c1] )
    ap3.start( [c1] )
    ap4.start( [c1] )
    ap5.start( [c1] )
    ap6.start( [c1] )
    ap7.start( [c1] )
    ap8.start( [c1] )
    ap9.start( [c1] )


    """uncomment to plot graph"""
    net.plotGraph(max_x=120, max_y=120)

    net.startMobility(startTime=0)
    net.mobility(sta14, 'start', time=1, position='43,102,0')
    net.mobility(sta14, 'stop', time=101, position='60,102,0')

    net.stopMobility(stopTime=502)

    """association control"""
#    a=Sta(sta14,"sf",ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9)
#    a.associationcontrol(sta14,"sf",ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9)

    """Seed"""
    #net.seed(19)

    """random walking"""
    #net.startMobility(startTime=0, model='RandomDirection',max_x=120,max_y=120, min_v=1,max_v=1)

    print "*** Running CLI"
    CLI( net )
    A=[]
    second = sleeptime(0,0,0)
    while 1==1:
        time.sleep(second)
        associationcontrol(sta14,ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9 )
        new_rssi = [chanFunt(ap1,sta14),chanFunt(ap2,sta14),chanFunt(ap3,sta14),
                    chanFunt(ap4,sta14),chanFunt(ap5,sta14),chanFunt(ap6,sta14),
                    chanFunt(ap7,sta14),chanFunt(ap8,sta14),chanFunt(ap9,sta14)]
        #print new_rssi
        #the_tag = rssi_tag(sta14)
        #print the_tag 
        #t1 = threading.Thread(target=iperf,args=([sta14,h1],'TCP',0.001,))
        #t2 = threading.Thread(target=tcpexamine,args = ([h1]))
        #t1.start()
        #t2.start()
        print (iperf([sta14,h1],l4Type= 'TCP',seconds=0.0000001,port=5001),datetime.datetime.now())
        #A.append(iperf([sta14,h1],l4Type='TCP',seconds=0.00001,port=5001))
        #if sta14.params['position'] == ('101.00','26.00','0.00'):
        #    print "*************reach destination"
        #if len(A)%1 == 0:
            #B = [float(i) for i in A]
        #    level = average([float(i)for i in A[-1:]])
        #    print "average:{0},{1}".format(level,datetime.datetime.now())
            #with open ("snake7.py","a+") as the_file:
            #    the_file.write(str(level))
            #    the_file.write(' ')
            #    the_file.write('\n')
            #    if len(A)%500==0:
            #        A=[]
            #        print "********************completed epoch{0}***************************".format(len(A)/500)
                
        
        #if new_rssi.count(0)==5 or new_rssi.count(0)==6:
        #    the_tag = rssi_tag(sta14)
        #    print new_rssi,the_tag
       
       #     with open ("tt.py","a+") as the_file:
       #        for item in new_rssi:
       #             the_file.write(str(item))
       #             the_file.write(" ")

            #the_file.write(":")
       #         the_file.write(str(the_tag))
       #         the_file.write('\n')
        #print new_rssi
        #the_tag = rssi_tag(sta14)
        #print the_tag    
        #A.append(iperf([h1,sta14],l4Type='TCP',seconds=0.01,port=5001))
        #::if len(A)%20 == 0:
        #    B = [float(i) for i in A]
            #print "average:{}".format(average(B[-20:]))
                 
        #    with open ('throughput.py','a+') as the_file:
        #        for i in A[-10:]
        #            the_file.write(A)
        #            the_file.write('\n')
           
        #with open ("tt.py","a+") as the_file:
        #    for item in new_rssi:
        #        the_file.write(str(item))
        #        the_file.write(" ")

            #the_file.write(":")
        #    the_file.write(str(the_tag))
        #    the_file.write('\n')

    print "*** Stopping network"
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    topology()

