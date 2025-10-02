#!/usr/bin/python3
import rclpy
import gym
from rclpy.node import Node
from rclpy.qos import QoSProfile,HistoryPolicy,ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from tello_msgs.srv import TelloAction
from fiducial_vlam_msgs.msg import Observations
import random
import time
import numpy as np
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pickle

global qlearn

przyspieszenie = 1

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
            print("    Stan: "+str(state)+"  Akcja: "+str(action)+"  Brak tego zestawu w tablicy Q!")
        else:
            self.q[(state, action)] = (1-self.alpha)*oldv + self.alpha * value#1-alfa dopisane
        #print("st:"+str(state)+" act:"+str(action)+" rew:"+str(reward)+" val:"+str(value)+" oldv:"+str(oldv))
        #time.sleep(0.1)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

class Klient_Serwisu(Node):

    def __init__(self):
      super().__init__("klient_serwisu_tello_action")
      self.cli = self.create_client(TelloAction, '/tello_action')
      self.get_logger().info('Oczekiwanie na serwis /drone1/tello_action')
      while not self.cli.wait_for_service(timeout_sec=3):
        self.get_logger().info('Serwis niedostępny, oczekiwanie...')
      self.req = TelloAction.Request()

      #self.empty_req = Empty.Request()

    def send_request(self, a):
      self.req.cmd = str(a)
      self.future = self.cli.call_async(self.req)
      #print(self.req)
      rclpy.spin_until_future_complete(self, self.future)
      return self.future.result()
    
    def polecenie(self,komunikat,hz,timeout):
      czas = time.time()
      self.get_logger().info("Wysyłanie polecenia: "+str(komunikat))
      powodzenie = 0
      while True:
        response = self.send_request(str(komunikat))
        if response.rc == 1:
          powodzenie = 1
          break
        else:
          if (time.time()-czas >= timeout):
             self.get_logger().error("NIE WYKONANO POLECENIA")
             powodzenie = 2
             break
        time.sleep(1/hz)
      time.sleep(4)
      return powodzenie
    
    
    
    





class MyNode(Node):
  
  def __init__(self, client_cb_croup, timer_cb_group,manual_calls):
    super().__init__("Q_learn_GA_drone")
    profil_qos = QoSProfile(
      reliability=ReliabilityPolicy.BEST_EFFORT,
      history=HistoryPolicy.KEEP_LAST,
      depth=1
    )
    self.rate = self.create_rate(5*przyspieszenie)
    self.programator = 0
    self.fiduical_obs = [0,-1,-1,-1,-1,-1,-1,-1,-1] # id, x0, y0, x1, y1, x2, y2, x3, y3
    self.fiduical_utrata = 0
    self.fiduical_obs_stare = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
    self.odom = Odometry()
    

    self.get_logger().info("Uruchomiono węzeł Q_learn_GA_drone")
    self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 2)
    subskryber = self.create_subscription(Observations,'/fiducial_observations',self.fiducial_observations_callback,0, callback_group=client_cb_croup)
    #subskryber2= self.create_subscription(Odometry,"/drone1/odom",self.Odometry_callback,qos_profile=profil_qos)
    if not manual_calls:
      self.call_timer = self.create_timer(0.0000001, self.timer_cb, callback_group=timer_cb_group)
      self.call_timer_1sek = self.create_timer(1/przyspieszenie, self.timer_cb_1sek, callback_group=timer_cb_group)

  def fiducial_observations_callback(self,msg):
    #self.get_logger().info("Otrzymano Callback z fiducial_observations")
    temp = str (msg.observations)
    #print("------temp------------------")
    #print(temp)
    temp2 = temp[1:-1].split(sep="(")
    temp2.remove("fiducial_vlam_msgs.msg.Observation")
    temp2 = str(temp2[0]).split(sep=")")
    temp2.remove("")
    temp2 = str(temp2[0]).split(sep=", ")
    #print("------temp 2------------------")
    #print(temp2)
    for i in range(1,9):
      temp3 = str(temp2[i]).split(sep="=")
      self.fiduical_obs[i]=round(float(temp3[1]),0)
    #print("------fiduical obs------------------")
    #print(self.fiduical_obs)

  

  '''def Odometry_callback(self,msg):

    self.odom=msg
    x = self.odom.pose.pose.position.x
    y = self.odom.pose.pose.position.y
    z = self.odom.pose.pose.position.z
    vx = self.odom.twist.twist.linear.x
    vy = self.odom.twist.twist.linear.y
    vz = self.odom.twist.twist.linear.z
    #print("Odebrano_x:" + str(round(x,2))+"| y:"+str(round(y,2))+"| z:"+str(round(z,2))
    #    +"| vx:"  + str(round(vx,2))+"| vy:"+str(round(vy,2))+"| vz:"+str(round(vz,2)))'''
    
  def timer_cb(self):
    return 0
  
  def timer_cb_1sek(self):
    #print(self.fiduical_obs)
    czy_stare = 0
    for i in range(1,9):
      if(self.fiduical_obs[i] == self.fiduical_obs_stare[i]):
        czy_stare = czy_stare + 1
    if czy_stare >= 8:
      self.fiduical_obs = [0,-1,-1,-1,-1,-1,-1,-1,-1]
    self.fiduical_obs_stare = self.fiduical_obs.copy()
    #print(self.fiduical_obs_stare)
    
  def publikuj_cmd_vel(self,vx,vy,vz,vroll,vpitch,vyaw):
    
    cmd_vel = Twist()
    cmd_vel.linear.x = float(vx)
    cmd_vel.linear.y = float(vy)
    cmd_vel.linear.z = float(vz)
    cmd_vel.angular.x = float(vroll)
    cmd_vel.angular.y = float(vpitch)
    cmd_vel.angular.z = float(vyaw)
    self.publisher_.publish(cmd_vel)
    #self.get_logger().info("Opublikowano na cmd_vel: "+str(cmd_vel))



  def program(self,start_time):
    #print(str(time.time()))
    '''
    if((time.time()-start_time)>5 and self.programator == 0):
      self.get_logger().warn("Rozpoczęcie lotu")
      self.publikuj_cmd_vel(0.2,0.1,0,0,0,0)
      self.programator=1
    if((time.time()-start_time)>18 and self.programator == 1):
      self.publikuj_cmd_vel(0,0,0,0,0,0)
      self.get_logger().warn("Koniec działania skryptu")
      self.programator=2
      '''
    if((time.time()-start_time)>10/przyspieszenie and self.programator == 0):
      self.programator = 2
    return self.programator
  
  
class GazeboCircuit2TurtlebotLidarEnv():

    def __init__(self,my_node_ref,klient_serwisu_ref,executor_ref):
        # Launch the simulation with the given launchfile name
        #gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        self.my_node = my_node_ref
        self.klient_serwisu = klient_serwisu_ref
        self.executor = executor_ref
        self.action_space = spaces.Discrete(5) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.success_counter = 0
        self.target_x = 0
        self.target_y = 0
        self.wysokosc_startowa = 1.5
        self.podzielnik_xy = 40
        self.target_z = self.wysokosc_startowa
        

        self._seed()


    def funkcja_testowa(self):
      kglfsjhglfs = 0
      while True:
        czas = time.time()
        while(time.time()-czas<1):
          self.executor.spin_once()
          time.sleep(0.01)
        state,done,reward,status = self.discretize_observation(self.my_node.fiduical_obs, 5)
        kglfsjhglfs += 1
        print(kglfsjhglfs,state, done, reward, status)
    
    def oblicz_pozycje_aruco(self,obserwacja):
      pozycja_aruco = []#x,y,dl_x,dl_y,roll
      sr_x = 0
      sr_y = 0
      dl_x = 0
      dl_y = 0
      yaw = 0
      if (obserwacja[1] !=-1):
        sr_x = round((obserwacja[1]+obserwacja[5])/2,0)
        sr_y = round((obserwacja[2]+obserwacja[6])/2,0)
        dl_x = round(np.sqrt((obserwacja[3]-obserwacja[1])**2 + (obserwacja[4]-obserwacja[2])**2),-1)
        if (obserwacja[7]-obserwacja[1] !=0):
          if(obserwacja[7]-obserwacja[1])> 0:
            if(obserwacja[8]-obserwacja[2])>= 0:
              yaw = np.arctan(abs(obserwacja[8]-obserwacja[2])/abs(obserwacja[7]-obserwacja[1]))
            else:
               yaw = -np.arctan(abs(obserwacja[8]-obserwacja[2])/abs(obserwacja[7]-obserwacja[1]))
          else:
            if(obserwacja[8]-obserwacja[2])>=0:
              yaw = np.pi-np.arctan(abs(obserwacja[8]-obserwacja[2])/abs(obserwacja[7]-obserwacja[1]))
            else: 
               yaw = -np.pi+np.arctan(abs(obserwacja[8]-obserwacja[2])/abs(obserwacja[7]-obserwacja[1]))
        else:
          if(obserwacja[8]-obserwacja[2])>= 0:
            yaw = np.pi/2
          else: yaw = -np.pi/2
      sr_x2 = int(round(sr_x/self.podzielnik_xy,0))
      sr_y2 = int(round(sr_y/self.podzielnik_xy,0))
      pozycja_aruco.append(int(sr_x2 * self.podzielnik_xy))
      pozycja_aruco.append(int(sr_y2 * self.podzielnik_xy))
      pozycja_aruco.append(int(dl_x))
      return pozycja_aruco,yaw

    def discretize_observation(self,data,action):
      
      state = []
      status = 0
      self.executor.spin_once()
      state,yaw = self.oblicz_pozycje_aruco(data)

      done = False
      
      #pożądana pozycja znacznika dla wyskokości 0,48m i równo nad znacznikiem
      target_x = 160
      target_y = 120
      target_z = 70 # taki rozmiar znacznika dla 0.48m nad znacznikiem

      reward = 0
      err_x = target_x- state[0]
      err_y = target_y-state[1]
      err_z = target_z - state[2]
      #print("Error: "+ str(err_x)+", "+ str(err_y))
      

      reward = -600-(abs(err_x)+abs(err_y)+abs(err_z))
      if action == 4:
        reward = reward + 200
        if (abs(err_x) < 80 and abs(err_y) < 80):
          reward = reward + 300
      
      if data[1] == -1: 
        self.my_node.fiduical_utrata = self.my_node.fiduical_utrata + 1
        #print ("wykryto -1  |  "+ str(self.my_node.fiduical_utrata))
        if self.my_node.fiduical_utrata >= 1:
          reward = -10000
          status = 1
          done = True
      else:
        self.my_node.fiduical_utrata = 0
        if err_z <= 0:
            reward = 100000
            self.success_counter += 1
            self.my_node.get_logger().warn("ORZEL WYLADOWAL :) Licznik = "+str(self.success_counter))
            status = 2
            done = True
      #print (state)
    
      return state,done,reward,status



    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.my_node.fiduical_obs = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
        if action == 0: #FORWARD
          self.target_x += 0.25
          self.klient_serwisu.polecenie("forward 25",2,15)
        elif action == 1: #Back
          self.target_x -= 0.25
          self.klient_serwisu.polecenie("back 25",2,15)
        elif action == 2: #LEFT
          self.target_y += 0.25
          self.klient_serwisu.polecenie("left 25",2,15)
        elif action == 3: #RIGHT
          self.target_y -= 0.25
          self.klient_serwisu.polecenie("right 25",2,15)
        elif action == 4: #Down
          self.target_z -= 0.25
          self.klient_serwisu.polecenie("down 25",2,15)
        koniec = 0
        #response = self.klient_serwisu.send_request_gazebo_unpause()
        #while koniec == 0:
        #  potw_x = False
        #  potw_y = False
        #  potw_z = False
        #  time.sleep(0.01/przyspieszenie)
        #  self.executor.spin_once()
        #  err_x = self.target_x-self.my_node.odom.pose.pose.position.x
        #  err_y = self.target_y-self.my_node.odom.pose.pose.position.y
        #  err_z = self.target_z-self.my_node.odom.pose.pose.position.z
        #  #print(err_x,err_y,err_z)
        #  if err_x > 0.05: self.my_node.publikuj_cmd_vel(0.5,0,0,0,0,0)
        #  elif err_x < -0.05: self.my_node.publikuj_cmd_vel(-0.5,0,0,0,0,0)
        #  else: potw_x = True
        #  if err_y > 0.05: self.my_node.publikuj_cmd_vel(0,0.5,0,0,0,0)
        #  elif err_y < -0.05: self.my_node.publikuj_cmd_vel(0,-0.5,0,0,0,0)
        #  else: potw_y = True
        #  if err_z > 0.05: self.my_node.publikuj_cmd_vel(0,0,0.5,0,0,0)
        #  elif err_z < -0.05: self.my_node.publikuj_cmd_vel(0,0,-0.5,0,0,0)
        #  else: potw_z = True

        #  if ((potw_x == True and potw_y == True) and potw_z == True):
        #     koniec = 1
      

        
        #response = self.klient_serwisu.send_request_gazebo_pause()
        czas = time.time()
        while(self.my_node.fiduical_obs[1]==-1):
          self.executor.spin_once()
          time.sleep(0.01)
          if(time.time()-czas > 0.5): break
        
        
        
        state,done,reward,status = self.discretize_observation(self.my_node.fiduical_obs, action)
        
        

        #if not done:
        #    if action == 0:
        #        reward = 1
        #    else:
        #        reward = 1
        #else:
        #    reward = -200

        return state, reward, done, status

    def reset(self):
        self.my_node.fiduical_obs = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
        powodzenie = 0
        print("Zerowanie pozycji")
        if self.target_z > self.wysokosc_startowa :
          while self.target_z > (self.wysokosc_startowa + 2) :
            powodzenie = self.klient_serwisu.polecenie(str("down 200"),2,15)
            if powodzenie == 1: self.target_z = self.target_z - 2
          powodzenie = self.klient_serwisu.polecenie(str("down "+str(int(abs(self.target_z-self.wysokosc_startowa)*100))),2,15)
          if powodzenie == 1:self.target_z = self.wysokosc_startowa
        if self.target_z < self.wysokosc_startowa :
          while self.target_z < (self.wysokosc_startowa - 2) :
            powodzenie = self.klient_serwisu.polecenie(str("up 200"),2,15)
            if powodzenie == 1: self.target_z = self.target_z + 2
          powodzenie = self.klient_serwisu.polecenie(str("up "+str(int(abs(self.target_z-self.wysokosc_startowa)*100))),2,15)
          if powodzenie == 1:self.target_z = self.wysokosc_startowa

        if self.target_x > 0 :
          while self.target_x > 2 :
            powodzenie = self.klient_serwisu.polecenie(str("back 200"),2,15)
            if powodzenie == 1: self.target_x = self.target_x - 2
          powodzenie = self.klient_serwisu.polecenie(str("back "+str(int(abs(self.target_x*100)))),2,15)
          if powodzenie == 1: self.target_x = 0
        if self.target_x < 0 :
          while self.target_x < -2 :
            powodzenie = self.klient_serwisu.polecenie(str("forward 200"),2,15)
            if powodzenie == 1: self.target_x = self.target_x + 2
          powodzenie = self.klient_serwisu.polecenie(str("forward "+str(int(abs(self.target_x*100)))),2,15)
          if powodzenie == 1: self.target_x = 0

        if self.target_y > 0 :
          while self.target_y > 2 :
            powodzenie = self.klient_serwisu.polecenie(str("right 200"),2,15)
            if powodzenie == 1: self.target_y = self.target_y - 2
          powodzenie = self.klient_serwisu.polecenie(str("right "+str(int(abs(self.target_y*100)))),2,15)
          if powodzenie == 1: self.target_y = 0
        if self.target_y < 0 :
          while self.target_y < -2 :
            powodzenie = self.klient_serwisu.polecenie(str("left 200"),2,15)
            if powodzenie == 1: self.target_y = self.target_y + 2
          powodzenie = self.klient_serwisu.polecenie(str("left "+str(int(abs(self.target_y*100)))),2,15)
          if powodzenie == 1:self.target_y = 0
        
        print("Zerowanie pozycji koniec")
        self.my_node.fiduical_utrata = 0

        data = None
        while data is None:
            try:
                data = self.my_node.fiduical_obs
            except:
                self.my_node.get_logger().error("Cos jest nie tak z fiduical obs")
                pass
        czas = time.time()
        while(self.my_node.fiduical_obs[1]==-1):
          self.executor.spin_once()
          time.sleep(0.01)
          if(time.time()-czas > 5):
              self.my_node.get_logger().error("Brak tagu w obrazie")
              self.klient_serwisu.polecenie("land",10,60) 
              break

        state,yaw = self.oblicz_pozycje_aruco(self.my_node.fiduical_obs)
      
        print (state)
        reset_aruco_x = 180 #70
        reset_aruco_y = 120 #210
        reset_aruco_yaw = 0
        #do przodu x rośnie na obrazie
        #w lewo y maleje na obrazie
        #tello_action.polecenie("rc 100 0 0 0",1,15) #zapierdala w prawo
        #tello_action.polecenie("rc 0 50 0 0",1,15)#zapierdala do przodu
        '''print("Stabilizacja pozycji")
        if state[0]>0:
          if yaw - reset_aruco_yaw > 0.1: 
            self.klient_serwisu.polecenie("rc 0 0 0 -25",10,60)
            while True:
              self.executor.spin_once()
              state,yaw = self.oblicz_pozycje_aruco(self.my_node.fiduical_obs)
              if yaw - reset_aruco_yaw <= 0.1:
                self.klient_serwisu.polecenie("stop",10,60)
                break
              if state[0] <= 0:
                self.klient_serwisu.polecenie("land",10,60)
                print("Brak tagu aruco w obrazie")
                break
              time.sleep(0.05)

          if yaw - reset_aruco_yaw < -0.1: 
            self.klient_serwisu.polecenie("rc 0 0 0 25",10,60)
            while True:
              self.executor.spin_once()
              state,yaw = self.oblicz_pozycje_aruco(self.my_node.fiduical_obs)
              if yaw - reset_aruco_yaw >= -0.1:
                self.klient_serwisu.polecenie("stop",10,60)
                break
              if state[0] <= 0:
                self.klient_serwisu.polecenie("land",10,60)
                print("Brak tagu aruco w obrazie")
                break
              time.sleep(0.05)


          if int(state[0]) - reset_aruco_x > 10: #trzeba do tyłu
            self.klient_serwisu.polecenie("rc 0 -30 0 0",10,60)
            while True:
              self.executor.spin_once()
              state,yaw = self.oblicz_pozycje_aruco(self.my_node.fiduical_obs)
              if int(state[0]) - reset_aruco_x <= 10:
                self.klient_serwisu.polecenie("stop",10,60)
                break
              if state[0] <= 0:
                self.klient_serwisu.polecenie("land",10,60)
                print("Brak tagu aruco w obrazie")
                break
              time.sleep(0.05)

          if int(state[0]) - reset_aruco_x < -10: # trzeba do przodu
            self.klient_serwisu.polecenie("rc 0 30 0 0",10,60)
            while True:
              self.executor.spin_once()
              state,yaw = self.oblicz_pozycje_aruco(self.my_node.fiduical_obs)
              if int(state[0]) - reset_aruco_x >= -10:
                self.klient_serwisu.polecenie("stop",10,60)
                break
              if state[0] <= 0:
                self.klient_serwisu.polecenie("land",10,60)
                print("Brak tagu aruco w obrazie")
                break
              time.sleep(0.05)

          if int(state[1]) - reset_aruco_y > 10: # trzeba w lewo
            self.klient_serwisu.polecenie("rc -30 0 0 0",10,60)
            while True:
              self.executor.spin_once()
              state,yaw = self.oblicz_pozycje_aruco(self.my_node.fiduical_obs)
              if int(state[1]) - reset_aruco_y <= 10:
                self.klient_serwisu.polecenie("stop",10,60)
                break
              if state[1] <= 0:
                self.klient_serwisu.polecenie("land",10,60)
                print("Brak tagu aruco w obrazie")
                break
              time.sleep(0.05)

          if int(state[1]) - reset_aruco_y < -10: # trzeba w prawo
            self.klient_serwisu.polecenie("rc 30 0 0 0",10,60)
            while True:
              self.executor.spin_once()
              state,yaw = self.oblicz_pozycje_aruco(self.my_node.fiduical_obs)
              if int(state[1]) - reset_aruco_y >= -10:
                self.klient_serwisu.polecenie("stop",10,60)
                break
              if state[1] <= 0:
                self.klient_serwisu.polecenie("land",10,60)
                print("Brak tagu aruco w obrazie")
                break
              time.sleep(0.05)

        else: print("Brak tagu aruco w obrazie")
        print("Stabilizacja pozycji koniec")'''
             
          
          


        return state



    
def main(args=None):

  try:
    global qlearn
    
    rclpy.init(args=args)
    tello_action = Klient_Serwisu()
    node = MyNode(client_cb_croup=ReentrantCallbackGroup(), timer_cb_group=ReentrantCallbackGroup(), manual_calls=False)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    
    
    
    
    #node.get_logger().info("Stan drona: "+str(response.OK))
    #env = gym.make('GazeboCircuit2TurtlebotLidar-v0')

    #outdir = '/tmp/gazebo_gym_experiments'
    env = GazeboCircuit2TurtlebotLidarEnv(node,tello_action,executor)
    
    last_time_steps = np.ndarray(0)

    qlearn = QLearn(actions=range(env.action_space.n),alpha=1, gamma=0.02, epsilon=0)


    try:
      qlearn.q=pickle.load(open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/Q", "rb" )) 
      node.get_logger().warn("Wczytano plik z obiektem Q")
    except:
      node.get_logger().error("Brak pliku z obiektem Q")

    initial_epsilon = qlearn.epsilon

    epsilon_discount = 1/45

    start_time = time.time()
    total_episodes = 1

    env.wysokosc_startowa = 1.75
    env.target_x = 0
    env.target_y = 0
    env.target_z = env.wysokosc_startowa

    tello_action.polecenie("downvision 1",1,5)
    tello_action.polecenie("takeoff",1,15)
    tello_action.polecenie("up "+str(int(env.wysokosc_startowa*100 - 100)),1,15)
    
    
    programator_wartosc = 0
    while rclpy.ok() and programator_wartosc < 2:
      programator_wartosc = node.program(start_time)
      executor.spin_once()
    

    fig, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    #plt.ion()
    vec_x = []
    cumulated_rewards = []
    max_rewards = []
    filtr_cumulated=[]

    #observation = env.reset()
    
    for x in range(total_episodes):
      done = False
      
      cumulated_reward = 0 
      highest_reward = -1000000000
      
      observation = env.reset()
      
      state = ''.join(map(str, observation))
      
      licznik = 0

      for i in range(50):
          licznik +=1
          action = qlearn.chooseAction(state)
          executor.spin_once()
          observation, reward, done, status = env.step(action)
          nextState = ''.join(map(str, observation))
          #print(nextState)

          #qlearn.learn(state, action, reward, nextState)
          #print(state,action,reward,nextState)
          cumulated_reward += reward
          
          if highest_reward < reward:
            highest_reward = reward
          if not(done):
              state = nextState
          else:
              last_time_steps = np.append(last_time_steps, [int(i + 1)])
              break
      tello_action.polecenie("land",10,60)
      mean_reward = round(cumulated_reward / licznik,0)

      vec_x.append(x)
      cumulated_rewards.append(mean_reward)

      m, s = divmod(int(time.time() - start_time), 60)
      h, m = divmod(m, 60)
      print ("EP: "+str(x+1)+" [a: "+str(round(qlearn.alpha,2))+" g: "
              +str(round(qlearn.gamma,2))+" e: "+str(round(qlearn.epsilon,2))
              +"]Rew:"+str(round(cumulated_reward,0))
              +"|Iter:"+str(licznik)
              +"|Mean rew:"+str(mean_reward)
              +"|MAX:"+str(highest_reward)
              +"|Time:%d:%02d:%02d" % (h, m, s))
      break


  finally:
    tello_action.polecenie("land",10,60)

    
#filtr_cumulated
if __name__ == '__main__':
  main()
