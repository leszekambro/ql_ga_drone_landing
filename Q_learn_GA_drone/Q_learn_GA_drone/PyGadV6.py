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
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import tf_transformations
import random
import time
import numpy as np
import pygad
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt

from matplotlib import style
import pickle


przyspieszenie = 1
global tello_action


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
    
    def set_entity_state(self,model_name, pose_x,pose_y,pose_z,pose_yaw):
      q= tf_transformations.quaternion_from_euler(0,0,pose_yaw,'ryxz')
      self.Set_entity_state.state = EntityState()
      self.Set_entity_state.state.name = model_name
      self.Set_entity_state.state.pose.position.x = float(pose_x)
      self.Set_entity_state.state.pose.position.y = float(pose_y)
      self.Set_entity_state.state.pose.position.z = float(pose_z)
      self.Set_entity_state.state.pose.orientation.x = q[0]
      self.Set_entity_state.state.pose.orientation.y = q[1]
      self.Set_entity_state.state.pose.orientation.z = q[2]
      self.Set_entity_state.state.pose.orientation.w = q[3]
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

    self.target_x = 160
    self.target_y = 120
    self.target_z = 80
    self.target_yaw = 0

    self.num_genes = 8
    self.num_generations = 50 # Number of generations.
    self.sol_per_pop = 50 # Number of solutions in the population.
    self.num_parents_mating = int(0.25 * self.sol_per_pop) # Number of solutions to be selected as parents in the mating pool.
    self.min_range = -5
    self.max_range = 5
    self.min_random_drop = -5
    self.max_random_drop = 5
    self.mutation_propability = 0.1
    self.crosover_propability = 0.3
    self.parent_selection_type="sss"
    self.crossover_type="single_point"
    self.ilosc_mutowanych_genow = 5


    self.rate = self.create_rate(5*przyspieszenie)
    self.programator = 0
    self.fiduical_obs = [0,-1,-1,-1,-1,-1,-1,-1,-1] # id, x0, y0, x1, y1, x2, y2, x3, y3
    self.fiduical_utrata = 0
    self.fiduical_obs_stare = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
    self.czas_poprzedni = time.time()
    self.dt = 0
    self.ux = 0
    self.uy = 0
    self.uz = 0
    self.uyaw = 0
    self.I_x = 0
    self.I_y = 0
    self.I_z = 0
    self.I_yaw = 0
    self.vx = 0
    self.vy = 0
    self.vz = 0
    self.vyaw = 0
    self.vx_poprzedni = 0
    self.vy_poprzedni = 0
    self.vz_poprzedni = 0
    self.vyaw_poprzedni = 0
    self.proba=0
    self.fig, self.ax1 = plt.subplots()
    self.solutions = []
    self.solutions_vecx=[]
    self.pozycja_x = 0
    self.pozycja_y = 0
    self.pozycja_z = 0


    self.last_fitness = 0
    self.sE = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.smE = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    self.start_time = time.time()
    self.end_time = time.time()
    self.odom = Odometry()

    self.get_logger().info("Uruchomiono węzeł Autonomia drona")
    self.publisher_ = self.create_publisher(Twist, '/drone1/cmd_vel', 2)
    subskryber = self.create_subscription(Observations,'/fiducial_observations',self.fiducial_observations_callback,1, callback_group=client_cb_croup)
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
  
  def oblicz_pozycje_aruco(self):
    sr_x = 0
    sr_y = 0
    dl_x = 0
    yaw = 0
    if (self.fiduical_obs[1] !=-1):
      sr_x = round((self.fiduical_obs[1]+self.fiduical_obs[5])/2,0)
      sr_y = round((self.fiduical_obs[2]+self.fiduical_obs[6])/2,0)
      dl_x = round(np.sqrt((self.fiduical_obs[3]-self.fiduical_obs[1])**2 + (self.fiduical_obs[4]-self.fiduical_obs[2])**2),0)
      if (self.fiduical_obs[7]-self.fiduical_obs[1] !=0):
        if(self.fiduical_obs[7]-self.fiduical_obs[1])> 0:
          if(self.fiduical_obs[8]-self.fiduical_obs[2])>= 0:
            yaw = np.arctan(abs(self.fiduical_obs[8]-self.fiduical_obs[2])/abs(self.fiduical_obs[7]-self.fiduical_obs[1]))
          else:
              yaw = -np.arctan(abs(self.fiduical_obs[8]-self.fiduical_obs[2])/abs(self.fiduical_obs[7]-self.fiduical_obs[1]))
        else:
          if(self.fiduical_obs[8]-self.fiduical_obs[2])>=0:
            yaw = np.pi-np.arctan(abs(self.fiduical_obs[8]-self.fiduical_obs[2])/abs(self.fiduical_obs[7]-self.fiduical_obs[1]))
          else: 
              yaw = -np.pi+np.arctan(abs(self.fiduical_obs[8]-self.fiduical_obs[2])/abs(self.fiduical_obs[7]-self.fiduical_obs[1]))
      else:
        if(self.fiduical_obs[8]-self.fiduical_obs[2])>= 0:
          yaw = np.pi/2
        else: yaw = -np.pi/2
    return sr_x,sr_y,dl_x,yaw

  
  def uchyb(self):
    ux=self.ux
    uy=self.uy
    uz=self.uz
    uyaw = self.uyaw
    sr_x,sr_y,dl_x,yaw = self.oblicz_pozycje_aruco()
    if sr_x > 0:
      ux = self.target_x - sr_x
      uy = self.target_y - sr_y
      uz = self.target_z - dl_x
      uyaw = self.target_yaw - yaw
      #print(ux)

    self.pozycja_x = self.odom.pose.pose.position.x
    self.pozycja_y = self.odom.pose.pose.position.y
    self.pozycja_z = self.odom.pose.pose.position.z
    return ux,uy,uz,uyaw
  
  def regulator_PID(self,kpx,kix,kdx,kpy,kiy,kdy,kpz,kiz,kdz,kpyaw,kiyaw,kdyaw):
    vx=0
    vy=0
    vz=0
    vyaw = 0
    ux_poprzedni = self.ux
    uy_poprzedni = self.uy
    uz_poprzedni = self.uz
    uyaw_poprzedni = self.uyaw
    self.ux,self.uy,self.uz,self.uyaw = self.uchyb()
    self.dt = (time.time()-self.czas_poprzedni)*(przyspieszenie)
    self.czas_poprzedni = time.time()
    #sterowanie po X
    P_x = self.ux*kpx
    self.I_x += self.ux * kix * self.dt
    D_x = kdx*(self.ux-ux_poprzedni)/self.dt 
    vx= P_x + self.I_x + D_x

    #sterowanie po Y
    P_y = self.uy*kpy
    self.I_y += self.uy * kiy * self.dt
    D_y = kdy*(self.uy-uy_poprzedni)/self.dt 
    vy= P_y + self.I_y + D_y

    #sterowanie po Z
    P_z = self.uz*kpz
    self.I_z += self.uz * kiz * self.dt
    D_z = kdz*(self.uz-uz_poprzedni)/self.dt 
    vz= P_z + self.I_z + D_z

    #sterowanie po Yaw
    P_yaw = self.uyaw*kpyaw
    self.I_yaw += self.uyaw * kiyaw * self.dt
    D_yaw = kdyaw*(self.uyaw-uyaw_poprzedni)/self.dt 
    vyaw= P_yaw + self.I_yaw + D_yaw

    #ograniczenia

    vx_max = 1
    vy_max = 1
    vz_max = 1
    vyaw_max = 1

    if(vx>vx_max): vx = float(vx_max)
    if(vx<-1*vx_max): vx = -1*float(vx_max)

    if(vy>vy_max): vy = vy_max
    if(vy<-1*vy_max): vy = -1*vy_max

    if(vz>vz_max): vz = vz_max
    if(vz<-1*vz_max): vz = -1*vz_max

    if(vyaw>vyaw_max): vyaw = vyaw_max
    if(vyaw<-1*vyaw_max): vyaw = -1*vyaw_max

    return vx,vy,vz,vyaw
  
  

  
  def reset(self):
        global tello_action
        # Resets the state of the environment and returns an initial observation.
        response = tello_action.send_request_gazebo_reset()
        # Unpause simulation to make observation
        random_x = 0
        random_y = 0
        random_z = 2.5
        random_yaw = 1
        response = tello_action.set_entity_state("tello_1",random_x,random_y,random_z,random_yaw)
        self.fiduical_obs = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
        self.fiduical_utrata = 0
        self.sE = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        response = tello_action.send_request_gazebo_unpause()
        #read data
        data = None
        while data is None:
            try:
                #data = rclpy.wait_for_message('/scan', LaserScan, timeout=5)

                data = self.fiduical_obs
            except:
                self.get_logger().error("Brak danych z fiduical_obs")
                pass
           
        #response = self.klient_serwisu.send_request_gazebo_pause()

        czas = time.time()
        while(time.time()-czas < 0.1):
          self.executor.spin_once()
          time.sleep(0.01)
          if(time.time()-czas > 0.5): break

        return 0
    
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

  def Ocena_jakosci(self, koniec,iter):
    
    self.vx = self.odom.twist.twist.linear.x
    self.vy = self.odom.twist.twist.linear.y
    self.vz = self.odom.twist.twist.linear.z
    self.vyaw = self.odom.twist.twist.angular.z
    ax = (self.vx-self.vx_poprzedni)/self.dt
    ay = (self.vy-self.vy_poprzedni)/self.dt
    az = (self.vz-self.vz_poprzedni)/self.dt
    ayaw = (self.vyaw-self.vyaw_poprzedni)/self.dt
    v_cel = 0.25 #m/s
    a_cel = 0#m/s2
    
    e = [abs(self.ux),abs(self.uy),abs(self.uz),abs(self.uyaw),abs(abs(self.vx)-v_cel),abs(abs(self.vy)-v_cel),abs(abs(self.vz)-v_cel),abs(abs(self.vyaw)-0),abs(abs(ax)-a_cel),abs(abs(ay)-a_cel),abs(abs(az)-a_cel),abs(abs(ayaw)-0),koniec]
    if(self.ux + self.uy)>10:#wyłączenie naliczania dla prędkości
      self.sE = [self.sE[0],self.sE[1],self.sE[2],self.sE[3],self.sE[4]+e[4],self.sE[5]+e[5],self.sE[6],self.sE[7],self.sE[8],self.sE[9],self.sE[10],self.sE[11],self.sE[12]]
    if(self.uyaw)>0.1:
      self.sE = [self.sE[0],self.sE[1],self.sE[2],self.sE[3],self.sE[4],self.sE[5],self.sE[6],self.sE[7]+e[7],self.sE[8],self.sE[9],self.sE[10],self.sE[11],self.sE[12]]
    
    self.sE = [e[0],e[1],e[2],self.sE[3]+e[3],self.sE[4],self.sE[5],self.sE[6]+e[6],self.sE[7],self.sE[8]+e[8],self.sE[9]+e[9],self.sE[10]+e[10],self.sE[11]+e[11],e[12]]
    
    self.smE = [round(abs(self.ux),2),round(abs(self.uy),2),round(abs(self.uz),2),round(abs(self.sE[3]/iter),2),
                round(self.sE[4]/iter,2),round(self.sE[5]/iter,2),round(self.sE[6]/iter,2),round(self.sE[7]/iter,2),
                round(self.sE[8]/iter,2),round(self.sE[9]/iter,2),round(self.sE[10]/iter,2),round(self.sE[11]/iter,2),
                round(self.sE[12],0)]
    self.vx_poprzedni = self.vx
    self.vy_poprzedni = self.vy
    self.vz_poprzedni = self.vz
    self.vyaw_poprzedni = self.vyaw
    return self.smE

  def Symulacja(self,solution):
    self.reset()
    koniec = 0
    iter = 0
    while (koniec == 0):
      czekaj = time.time()
      iter += 1
      self.fiduical_obs = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
      while (time.time()-czekaj < 1/przyspieszenie):
        self.executor.spin_once()
        if(self.fiduical_obs[1]>0):break
        #time.sleep(0.1/przyspieszenie)

      vx,vy,vz,vyaw =self.regulator_PID(solution[0]*0.01,0,solution[1]*0.001,
                                        solution[2]*0.01,0,solution[3]*0.001,
                                        solution[4]*0.01,0,solution[5]*0.001,
                                        solution[6]*1,0,solution[7]*1)
      self.publikuj_cmd_vel(vx,vy,vz,0,0,vyaw)
      
      if (self.uz < 5 and self.uz > -40): koniec = -1
      if (self.fiduical_obs[1]<0): koniec = 1
      if (self.uz <= -40) : koniec = 3
      sE = self.Ocena_jakosci(koniec,iter)
    
    self.publikuj_cmd_vel(0,0,0,0,0,0)
    
    
    return sE,iter

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
  

  def fitness_func(self,ga_instance, solution, solution_idx):
      #self.sE = etap_3.Dron2D(solution, p_zad, 0,50,0)
      self.proba +=1
      sE,iter = self.Symulacja(solution)
      #e = [abs(self.ux),abs(self.uy),abs(self.uz),abs(self.uyaw),
      #   abs(self.vx),abs(self.vy),abs(self.vz),abs(self.vyaw),
      #   abs(ax),abs(ay),abs(az),abs(ayaw),koniec]
      fitness = 50000 - (100*abs(sE[0]) + 100*abs(sE[1]) + 500*abs(sE[2]) + 2500*abs(sE[3]) + 5000*abs(sE[4]) + 5000* abs(sE[5])+ 5000* abs(sE[6]) + 5000* abs(sE[7]) + 10000* abs(sE[8])+ 10000* abs(sE[9])+ 10000* abs(sE[10])+ 10000* abs(sE[11])+ 50000* sE[12])
      #Teoretyczny max to 100k
      #max: 100* 320 + 100* 240 + 500* 200 + 1000* 1 + 1000* 1 + 1000* 1+ 1000* 1 +1000 * 1 + 2000 * 1 + 2000 * 1 + 2000* 1 + 2000 * 1 + 50k * 3= 318k
      #     50k - 318 k = -268 k - teoretyczne minimum
      print ("Pr:"+str(self.proba)+" |sE:"+str(sE)+" |Iter:"+ str(iter)+" |Fitness:"+str(int(fitness))+"       ",end='\r')
      #print (sE)
      #print (fitness)
      print("Pozycja lądowania: "+str(round(self.pozycja_x,2)) + " | " +str(round(self.pozycja_y,2))  + " | "+str(round(self.pozycja_z,2)) )
      return fitness



  def on_mutation(self,ga_instance,offspring_m):
      for i in range(0,self.ilosc_mutowanych_genow):
          a=(self.max_range-self.min_range)/2
          ga_instance.last_generation_offspring_mutation[-i][:]=[2*a*random.random()-a, 2*a*random.random()-a, 2*a*random.random()-a,2*a*random.random()-a, 2*a*random.random()-a, 2*a*random.random()-a, 2*a*random.random()-a, 2*a*random.random()-a]


  def on_generation(self,ga_instance):
      
      t_od_start = (time.time() - self.start_time)
      t_gen = t_od_start / ga_instance.generations_completed
      t_est = (self.num_generations - ga_instance.generations_completed) * t_gen
      solution, solution_fitness, solution_idx = ga_instance.best_solution()
      self.solutions.append(solution_fitness)
      self.solutions_vecx.append(ga_instance.generations_completed)
      self.ax1.clear()
      self.ax1.grid(1)
      self.ax1.plot(self.solutions_vecx, self.solutions,linewidth=2)
      self.ax1.set(xlabel='Numer generacji', ylabel='Wartość funkcji dopasowania',title='Wykres najlepszej wartości funkcji dopasowania')
      plt.pause(0.01)
      m_od_start,s_od_start=divmod(int(t_od_start),60)
      h_od_start,m_od_start = divmod(m_od_start,60)
      m_gen,s_gen = divmod(int(t_gen),60)
      h_gen,m_gen = divmod(m_gen,60)
      m_est,s_est=divmod(int(t_est),60)
      h_est,m_est=divmod(m_est,60)
      print(end='\x1b[2K')
      print("------------------")
      print(f"Generation = " + str(ga_instance.generations_completed) + " / " + str(self.num_generations) + "  ("
            + str(round(ga_instance.generations_completed/self.num_generations*100,1)) +"%)")
      print(f"Best solution:    = {solution}")
      #print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
      #print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - self.last_fitness}")
      print(f"Fitness    = {solution_fitness}")
      print(f"Change     = {solution_fitness - self.last_fitness}")
      print("Czas od startu:           " + str(round(t_od_start,2))+" s (%02d:%02d:%02d)" % (int(h_od_start), int(m_od_start), int(s_od_start)))
      print("Sredni czas generacji:    " + str(round(t_gen,2))+" s (%02d:%02d:%02d)" % (int(h_gen), int(m_gen), int(s_gen)))
      print("Do zakonczenia zostalo:   " + str(round(t_est,0))+" s (%02d:%02d:%02d)" % (int(h_est), int(m_est), int(s_est)))
      print()
      self.last_fitness = solution_fitness
      self.proba=0
    
    
def main(args=None):

  try:
    global tello_action
    rclpy.init(args=args)
    tello_action = Klient_Serwisu()
    node = MyNode(client_cb_croup=ReentrantCallbackGroup(), timer_cb_group=ReentrantCallbackGroup(), manual_calls=False)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    response = tello_action.send_request(str("takeoff"))



    
  
    start_time = time.time()
    programator_wartosc = 0
    while rclpy.ok() and programator_wartosc < 2:
      programator_wartosc = node.program(start_time)
      executor.spin_once()
    node.get_logger().info("Rozpoczeto nauczanie")
    node.reset()

    #testowo:
    #nastawy_testowe = []
    #node.Symulacja(nastawy_testowe)
    #print("sE: "+ str(node.smE))
    #while True:
    #  time.sleep(1000)
    

    ga_instance = pygad.GA(num_generations=node.num_generations,
                        num_parents_mating=node.num_parents_mating,
                        parent_selection_type=node.parent_selection_type,
                        crossover_type=node.crossover_type,
                        sol_per_pop=node.sol_per_pop,
                        num_genes=node.num_genes,
                        fitness_func=node.fitness_func,
                        init_range_low=node.min_range,
                        init_range_high=node.max_range,
                        random_mutation_min_val=node.min_random_drop,
                        random_mutation_max_val=node.max_random_drop,
                        mutation_probability=node.mutation_propability,
                        crossover_probability=node.crosover_propability,
                        save_solutions=True,
                        #parallel_processing=["process", 4])
                        on_mutation = node.on_mutation,
                        on_generation=node.on_generation)

    
    node.start_time = time.time()
    ga_instance.run()
    
    print("!!!")
    node.get_logger().info("Koniec działania algorytmu")
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    pickle.dump(node.solutions,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/PyGad_solutions", "wb" ))
    pickle.dump(node.solutions_vecx,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/PyGad_solutions_vecx", "wb" ))
    pickle.dump(solution,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/PyGad_solution", "wb" ))
    while True:
      time.sleep(1000)
    
  finally:
    node.publikuj_cmd_vel(0,0,0,0,0,0)
    node.end_time = time.time()
    print("Czas działania algorytmu: " + str(round(node.end_time-node.start_time,2)) + " s")

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    pickle.dump(node.solutions,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/PyGad_solutions", "wb" ))
    pickle.dump(node.solutions_vecx,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/PyGad_solutions_vecx", "wb" ))
    pickle.dump(solution,open("/home/[your_username]/tello_ros_ws/src/Q_learn_GA_drone/files/PyGad_solution", "wb" ))
    print(f"Parametry najlepszego rozwiązania : {solution}")
    print(f"Najlepsza wartość fitness function = {solution_fitness}")
    print(f"Idx:  = {solution_idx}")
    ga_instance.plot_fitness()
    ga_instance.plot_new_solution_rate()
    #node.destroy_node()
    #tello_action.destroy_node()
    #rclpy.shutdown()

    

if __name__ == '__main__':
  main()
