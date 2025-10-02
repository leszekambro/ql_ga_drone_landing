#!/usr/bin/python3
import rclpy
import gym
from rclpy.node import Node
from rclpy.qos import QoSProfile,HistoryPolicy,ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
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
from mpl_toolkits.mplot3d import Axes3D

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
        #print(q,maxQ,i)

        action = self.actions[i]        
        if return_q:
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

class Klient_Serwisu(Node):

    def __init__(self):
      super().__init__("klient_serwisu_tello_action")
      self.cli = self.create_client(TelloAction, '/drone1/tello_action')
      self.get_logger().info('Oczekiwanie na serwis /drone1/tello_action')
      while not self.cli.wait_for_service(timeout_sec=3):
        self.get_logger().info('Serwis niedostępny, oczekiwanie...')
      self.req = TelloAction.Request()

      self.SET_ENTITY_STATE = self.create_client(SetEntityState, '/gazebo/set_entity_state')
      while not self.SET_ENTITY_STATE.wait_for_service(timeout_sec=3):
        self.get_logger().info('Serwis "/gazebo/set_entity_state" niedostępny, oczekiwanie...')
      self.Set_entity_state = SetEntityState.Request()

      self.unpause = self.create_client(Empty, '/unpause_physics')
      while not self.unpause.wait_for_service(timeout_sec=3):
        self.get_logger().info('Serwis /unpause_physics niedostępny, oczekiwanie...')
      self.pause = self.create_client(Empty,'/pause_physics')
      while not self.pause.wait_for_service(timeout_sec=3):
        self.get_logger().info('Serwis /pause_physics niedostępny, oczekiwanie...')
      self.reset = self.create_client(Empty,'/reset_world')
      while not self.reset.wait_for_service(timeout_sec=3):
        self.get_logger().info('Serwis /reset_world niedostępny, oczekiwanie...')
      self.empty_req = Empty.Request()

    def send_request(self, a):
      self.req.cmd = a
      self.future = self.cli.call_async(self.req)
      rclpy.spin_until_future_complete(self, self.future)
      return self.future.result()
    
    def send_request_gazebo_pause(self):
      self.future = self.pause.call_async(self.empty_req)
      rclpy.spin_until_future_complete(self, self.future)
      return self.future.result()
    
    def send_request_gazebo_unpause(self):
      self.future = self.unpause.call_async(self.empty_req)
      rclpy.spin_until_future_complete(self, self.future)
      return self.future.result()
    
    def send_request_gazebo_reset(self):
      self.future = self.reset.call_async(self.empty_req)
      rclpy.spin_until_future_complete(self, self.future)
      return self.future.result()
    
    def set_entity_state(self,model_name, pose_x,pose_y,pose_z):
      self.Set_entity_state.state = EntityState()
      self.Set_entity_state.state.name = model_name
      self.Set_entity_state.state.pose.position.x = float(pose_x)
      self.Set_entity_state.state.pose.position.y = float(pose_y)
      self.Set_entity_state.state.pose.position.z = float(pose_z)
      self.Set_entity_state.state.pose.orientation.x = 0.0
      self.Set_entity_state.state.pose.orientation.y = 0.0
      self.Set_entity_state.state.pose.orientation.z = 0.0
      self.Set_entity_state.state.pose.orientation.w = 1.0
      self.Set_entity_state.state.twist.linear.x = 0.0
      self.Set_entity_state.state.twist.linear.y = 0.0
      self.Set_entity_state.state.twist.linear.z = 0.0
      self.Set_entity_state.state.twist.angular.x = 0.0
      self.Set_entity_state.state.twist.angular.y = 0.0
      self.Set_entity_state.state.twist.angular.z = 0.0


      self.future = self.SET_ENTITY_STATE.call_async(self.Set_entity_state)
      rclpy.spin_until_future_complete(self, self.future)
      return self.future.result()

       
    
    





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
    

    self.get_logger().info("Uruchomiono węzeł Autonomia drona")
    self.publisher_ = self.create_publisher(Twist, '/drone1/cmd_vel', 2)
    subskryber = self.create_subscription(Observations,'/fiducial_observations',self.fiducial_observations_callback,0, callback_group=client_cb_croup)
    subskryber2= self.create_subscription(Odometry,"/drone1/odom",self.Odometry_callback,qos_profile=profil_qos)
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

  

  def Odometry_callback(self,msg):

    self.odom=msg
    x = self.odom.pose.pose.position.x
    y = self.odom.pose.pose.position.y
    z = self.odom.pose.pose.position.z
    vx = self.odom.twist.twist.linear.x
    vy = self.odom.twist.twist.linear.y
    vz = self.odom.twist.twist.linear.z
    #print("Odebrano_x:" + str(round(x,2))+"| y:"+str(round(y,2))+"| z:"+str(round(z,2))
    #    +"| vx:"  + str(round(vx,2))+"| vy:"+str(round(vy,2))+"| vz:"+str(round(vz,2)))
    
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
        self.target_z = 2.5
        self.podzielnik_xy = 10
        self.pozycja_x = 0
        self.pozycja_y = 0
        self.pozycja_z = 0

        self._seed()
    
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
        dl_x = round(np.sqrt((obserwacja[3]-obserwacja[1])**2 + (obserwacja[4]-obserwacja[2])**2),-1)#wariant nowy (dlugosc boku)
        #dl_x = round(np.sqrt((obserwacja[5]-obserwacja[1])**2 + (obserwacja[6]-obserwacja[2])**2),-1)#wariant stary (przekątna)
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
      #print(pozycja_aruco,yaw)
      return pozycja_aruco,yaw

    def discretize_observation(self,data,action):
      state = []
      status = 0
      self.executor.spin_once()
      state,yaw = self.oblicz_pozycje_aruco(data)

      done = False
      #pożądana pozycja znacznika dla wyskokości 0,48m i równo nad znacznikiem
      target_x = 160
      target_y = 140
      target_z = 80 # 120 - taki rozmiar znacznika dla 0.48m nad znacznikiem

      reward = 0
      err_x = target_x- state[0]
      err_y = target_y-state[1]
      err_z = target_z - state[2]
      #print("Error: "+ str(err_x)+", "+ str(err_y))
      
      #NAGRODA WARIANT 1
      '''
      if (abs(err_x) < 40 and 
          abs(err_y) < 40):
         reward = 40
      if action == 4:
         reward = reward + 10'''
      
      #drugi typ nagrody
      
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
        elif action == 1: #Back
          self.target_x -= 0.25
        elif action == 2: #LEFT
          self.target_y += 0.25
        elif action == 3: #RIGHT
          self.target_y -= 0.25
        elif action == 4: #Down
          self.target_z -= 0.25
        #elif action == 5: #Up
          #self.target_z += 0.25
        koniec = 0
        response = self.klient_serwisu.send_request_gazebo_unpause()
        while koniec == 0:
          potw_x = False
          potw_y = False
          potw_z = False
          time.sleep(0.01/przyspieszenie)
          self.executor.spin_once()
          self.pozycja_x = self.my_node.odom.pose.pose.position.x
          self.pozycja_y = self.my_node.odom.pose.pose.position.y
          self.pozycja_z = self.my_node.odom.pose.pose.position.z
          
          err_x = self.target_x-self.pozycja_x
          err_y = self.target_y-self.pozycja_y
          err_z = self.target_z-self.pozycja_z
          Wykres_trajektori_3D(self.pozycja_x,self.pozycja_y,self.pozycja_z)

          #print(err_x,err_y,err_z)
          if err_x > 0.05: self.my_node.publikuj_cmd_vel(0.5,0,0,0,0,0)
          elif err_x < -0.05: self.my_node.publikuj_cmd_vel(-0.5,0,0,0,0,0)
          else: potw_x = True
          if err_y > 0.05: self.my_node.publikuj_cmd_vel(0,0.5,0,0,0,0)
          elif err_y < -0.05: self.my_node.publikuj_cmd_vel(0,-0.5,0,0,0,0)
          else: potw_y = True
          if err_z > 0.05: self.my_node.publikuj_cmd_vel(0,0,0.5,0,0,0)
          elif err_z < -0.05: self.my_node.publikuj_cmd_vel(0,0,-0.5,0,0,0)
          else: potw_z = True

          if ((potw_x == True and potw_y == True) and potw_z == True):
             koniec = 1
      

        
        response = self.klient_serwisu.send_request_gazebo_pause()
        czas = time.time()
        while(self.my_node.fiduical_obs[1]==-1):
          self.executor.spin_once()
          time.sleep(0.01)
          if(time.time()-czas > 0.2): break
        
        
        
        state,done,reward,status = self.discretize_observation(self.my_node.fiduical_obs, action)

        return state, reward, done, status

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        response = self.klient_serwisu.send_request_gazebo_reset()
        self.my_node.fiduical_obs = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
        self.target_x = 2*random.random()
        self.target_y = 2*random.random()
        self.target_z = 3
        response = self.klient_serwisu.set_entity_state("tello_1",self.target_x,self.target_y,self.target_z)
        
        
        # Unpause simulation to make observation
        response = self.klient_serwisu.send_request_gazebo_unpause()

        self.my_node.fiduical_utrata = 0

        #read laser data
        data = None
        while data is None:
            try:
                #data = rclpy.wait_for_message('/scan', LaserScan, timeout=5)

                data = self.my_node.fiduical_obs
            except:
                self.my_node.get_logger().error("Brak danych z fiduical_obs")
                pass
            czas = time.time()
            while(self.my_node.fiduical_obs[1]==-1):
              self.executor.spin_once()
              time.sleep(0.01)
              if(time.time()-czas > 5): break
        response = self.klient_serwisu.send_request_gazebo_pause()

        state,done,reward,status = self.discretize_observation(data,5)

        return state,done,reward,status

#def render():
#    render_skip = 0 #Skip first X episodes.
#    render_interval = 50 #Show render Every Y episodes.
#    render_episodes = 10 #Show Z episodes every rendering
#    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
#        env.render()
#    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
#        env.render(close=True)




def Wykres_trajektori_3D(x,y,z):
   if not hasattr(Wykres_trajektori_3D, "trajektoria"):
      Wykres_trajektori_3D.trajektoria = {"x": [], "y": [], "z": []}
      
   Wykres_trajektori_3D.trajektoria["x"].append(x)
   Wykres_trajektori_3D.trajektoria["y"].append(y)
   Wykres_trajektori_3D.trajektoria["z"].append(z)

def Wyswietl_wykres_trajektorii_3D():
   traj = Wykres_trajektori_3D.trajektoria
   pickle.dump(traj,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/Trajektoria_3D", "wb" ))
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.plot(traj["x"], traj["y"], traj["z"])
   ax.set(xlabel='Position in X axis [m]', ylabel='Position in Y axis [m]',zlabel= 'Position in Z axis [m]'#title='Average reward value graph for the divisor test'
            )
   plt.axis('equal')
   plt.show()
   while True:
     time.sleep(1000)
   
    
    
def main(args=None):

  try:
    global qlearn
    
    rclpy.init(args=args)
    tello_action = Klient_Serwisu()
    node = MyNode(client_cb_croup=ReentrantCallbackGroup(), timer_cb_group=ReentrantCallbackGroup(), manual_calls=False)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    response = tello_action.send_request(str("takeoff"))
    node.get_logger().info("Stan drona: "+str(response.OK))
    env = GazeboCircuit2TurtlebotLidarEnv(node,tello_action,executor)
    
    last_time_steps = np.ndarray(0)

    qlearn = QLearn(actions=range(env.action_space.n),alpha=0.2, gamma=0.02, epsilon=0.5)


    try:
      qlearn.q=pickle.load(open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/Q", "rb" )) 
      node.get_logger().warn("Wczytano plik z obiektem Q")
    except:
      node.get_logger().error("Brak pliku z obiektem Q")

    epsilon = qlearn.epsilon
    total_episodes = 20000
    env.podzielnik_xy = 75
    epsilon_discount = 1.1/total_episodes

    start_time = time.time()
    
    
    programator_wartosc = 0
    while rclpy.ok() and programator_wartosc < 2:
      programator_wartosc = node.program(start_time)
      executor.spin_once()
    node.get_logger().info("Rozpoczeto nauczanie")

    

    fig, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    #plt.ion()
    vec_x = []
    cumulated_rewards = []
    max_rewards = []
    filtr_cumulated=[]
    observation = env.reset()
    pomiar_testowy = 0
    ilosc_testow = []
    sukcesy_test = []
    sukces_test = 0

    kiedy_test = 100
    dlugosc_testu = 10
    krotnosc = 0
    
    for x in range(total_episodes):
      done = False
      
      cumulated_reward = 0
      highest_reward = -1000000000
      
      observation,done2,reward2,status2 = env.reset()
      
      '''kglfsjhglfs = 0
      while True:
        czas = time.time()
        while(time.time()-czas<1):
          executor.spin_once()
          time.sleep(0.01)
        random_x = 0.2
        random_y = 0.2
        random_z = 2
        response = tello_action.set_entity_state("tello_1",random_x,random_y,random_z)
        time.sleep(0.1)
        state,done,reward,status = env.discretize_observation(node.fiduical_obs, 5)
        kglfsjhglfs += 1
        print(kglfsjhglfs,state, done, reward, status)'''
      
      if x%kiedy_test >= 0 and x%kiedy_test < dlugosc_testu:
         qlearn.epsilon = 0 
         if pomiar_testowy == 0:
            pomiar_testowy = 1
      else: 
        if pomiar_testowy > 0:
           ilosc_testow.append(round((x-dlugosc_testu)/kiedy_test,0))
           sukcesy_test.append(sukces_test)
           ax3.clear()
           ax3.grid(1)
           ax3.plot(ilosc_testow,sukcesy_test,linewidth=1)
           ax3.scatter(ilosc_testow, sukcesy_test,s=25)
           ax3.set(xlabel='Numer testu', ylabel='Ile razy wylądował',title='Wykres skuteczności lądowania podczas testu')
           pickle.dump(qlearn.q,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/plikiQ/Q"+str(x), "wb" ))
          
           plt.pause(0.01)
        pomiar_testowy = 0
        sukces_test = 0
        #if epsilon > 0.2:
        #  epsilon -= epsilon_discount
        #qlearn.epsilon = epsilon
        qlearn.epsilon = 0.7

      state = ''.join(map(str, observation))
      
      licznik = 0

      for i in range(50):
          licznik +=1
          # Pick an action based on the current state
          action = qlearn.chooseAction(state)

          # Execute the action and get feedback
          executor.spin_once()
          observation, reward, done, status = env.step(action)
          

          nextState = ''.join(map(str, observation))
          #print(nextState)

          qlearn.learn(state, action, reward, nextState)
          #print(state,action,reward,nextState)

          cumulated_reward += reward

          if pomiar_testowy > 0:
            if reward > 50000:
              sukces_test += 1
            
          if highest_reward < reward:
            highest_reward = reward
          #if status == 1:
          #  cumulated_reward +=100
            

          #env._flush(force=True)

          if not(done):
              state = nextState
          else:
              last_time_steps = np.append(last_time_steps, [int(i + 1)])
              break
      mean_reward = round(cumulated_reward / licznik,0)

      vec_x.append(x)
      cumulated_rewards.append(mean_reward)
      
      ilosc_probek_filtr = 500
      suma_filtr = 0
      ilosc_cumulated_max = len(cumulated_rewards)
      ilosc_cumulated_min = len(cumulated_rewards)-ilosc_probek_filtr
      if ilosc_cumulated_min < 0: ilosc_cumulated_min = 0
      for i in range(ilosc_cumulated_min,ilosc_cumulated_max):
        suma_filtr += cumulated_rewards[i]
      if (ilosc_cumulated_max-ilosc_cumulated_min) > 0: filtr = suma_filtr / (ilosc_cumulated_max-ilosc_cumulated_min)
      else: filtr = 0
      max_rewards.append(highest_reward)
      filtr_cumulated.append(filtr)

      if x%10==0:
        ax1.clear()
        ax2.clear()
        ax1.grid(1)
        ax2.grid(1)
        ax1.plot(vec_x, cumulated_rewards,linewidth=0.3)
        ax1.plot(vec_x, filtr_cumulated,linewidth=1)
        ax2.plot(vec_x,max_rewards,linewidth=0.5)
        ax1.set(xlabel='Liczba iteracji', ylabel='Średnia wartość nagrody',title='Wykres średniej wartości nagrody')
        ax2.set(xlabel='Liczba iteracji', ylabel='Max wartość nagrody',title='Wykres maksymalnej wartości nagrody')
        #plt.plot()
        plt.pause(0.01)

      m, s = divmod(int(time.time() - start_time), 60)
      h, m = divmod(m, 60)
      print ("EP: "+str(x+1)+"/"+str(total_episodes)+"("+str(round(100*(x+1)/total_episodes,2))+"%)"+" [a: "+str(round(qlearn.alpha,2))+" g: "
              +str(round(qlearn.gamma,2))+" e: "+str(round(qlearn.epsilon,2))
              +"]Rew:"+str(round(cumulated_reward,0))
              +"|Iter:"+str(licznik)
              +"|Mean rew:"+str(mean_reward)
              +"|MAX:"+str(highest_reward)
              +"|Time:%d:%02d:%02d" % (h, m, s))
      
      #Wyswietl_wykres_trajektorii_3D()
      print("Pozycja lądowania: "+str(round(env.pozycja_x,2)) + " | " +str(round(env.pozycja_y,2))  + " | "+str(round(env.pozycja_z,2))  + " | " )


      if pomiar_testowy > 0:
        print("Trwa test nauczania. Proba: "+str(pomiar_testowy)+ " Ilosc sukcesow: "+str(sukces_test))
        if pomiar_testowy == dlugosc_testu:
          node.get_logger().warn("Zakończono pomiar testowy.                                Skutecznosc ladowania: "+str(round((sukces_test/dlugosc_testu)*100,0))+"%")
          if(sukces_test/dlugosc_testu >= 1):
             krotnosc += 1
             if krotnosc == 3:
                print("Koniec nauczania. Osiągnięto próg skuteczności 100% trzykrotnie")
                break
          else: krotnosc = 0
            
        pomiar_testowy += 1
    ilosc_testow.append(round((x-dlugosc_testu)/kiedy_test,0))
    sukcesy_test.append(sukces_test)
    print("filtr cumulated ostatni: "+str(filtr_cumulated[len(filtr_cumulated)-1]))
    pickle.dump(qlearn.q,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/Q", "wb" ))
    pickle.dump(vec_x,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/vec_x", "wb" ))
    pickle.dump(cumulated_rewards,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/cumulated_rewards", "wb" ))
    pickle.dump(max_rewards,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/max_rewards", "wb" ))
    pickle.dump(filtr_cumulated,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/filtr_cumulated", "wb" ))
    pickle.dump(ilosc_testow,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/ilosc_testow", "wb" ))
    pickle.dump(sukcesy_test,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/sukcesy_test", "wb" ))    
    ax3.clear()
    ax3.grid(1)
    ax3.plot(ilosc_testow,sukcesy_test,linewidth=1)
    ax3.scatter(ilosc_testow, sukcesy_test,s=25)
      
    plt.show()
    while True:
      time.sleep(1000)


  finally:
    print("")
    print("Zapis do pliku")
    
    pickle.dump(qlearn.q,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/Q", "wb" ))
    pickle.dump(vec_x,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/vec_x", "wb" ))
    pickle.dump(cumulated_rewards,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/cumulated_rewards", "wb" ))
    pickle.dump(max_rewards,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/max_rewards", "wb" ))
    pickle.dump(filtr_cumulated,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/filtr_cumulated", "wb" ))
    pickle.dump(ilosc_testow,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/ilosc_testow", "wb" ))
    pickle.dump(sukcesy_test,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/sukcesy_test", "wb" ))
    node.destroy_node()
    tello_action.destroy_node()
    rclpy.shutdown()
#filtr_cumulated
if __name__ == '__main__':
  main()
