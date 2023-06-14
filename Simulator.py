import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import pandas as pd
import time


class Exchanger:
    def __init__(self,length,dictionary,zones,countercurrent,v_desv = False):
        self.storage = Storage()
        self.sensor = Sensor(self, v_desv)
        self.dictionary = dictionary
        self.zones = zones+2 #se añaden las dos zonas de los bordes
        self.total_length = length*((zones+2)/zones) #se añaden las dos zonas de los bordes de "fuera del sistema", se compensa la longitud para que los de dentro sean del mismo tamaño
        self.countercurrent = countercurrent
        self.controller = Controller(self,[self.dictionary['initial_t_entry_inside'],self.dictionary['initial_t_entry_outside']])
        self.sections = self.create_sections()
        self.controller.fix_boundary_temperatures()
        
    def create_sections(self):
        section_list = [ExchangerSection(self.dictionary, self.total_length/self.zones) for i in range(self.zones)]
        return section_list
    
    def get_T_in_dot(self,index):
        return (self.sections[index].w_in * self.sections[index].dictionary['cp_inside']* (self.sections[index-1].T_in - self.sections[index].T_in) - self.sections[index].get_q) / (self.sections[index].mass_inside * self.sections[index].dictionary['cp_inside'])
    
    def get_T_out_dot(self,index):
        if self.countercurrent:
            return (self.sections[index].w_out * self.sections[index].dictionary['cp_outside']* (self.sections[index+1].T_out - self.sections[index].T_out) + self.sections[index].get_q) / (self.sections[index].mass_outside * self.sections[index].dictionary['cp_outside'])
        return (self.sections[index].w_out * self.sections[index].dictionary['cp_outside']* (self.sections[index-1].T_out - self.sections[index].T_out) + self.sections[index].get_q) / (self.sections[index].mass_outside * self.sections[index].dictionary['cp_outside'])

    def integration(self):
        self.controller.fix_boundary_temperatures()
        if self.countercurrent :
            self.sections[0].T_out += self.get_T_out_dot(0) * DT
        for index in range(1,len(self.sections)):
            
            self.sections[index].T_in += self.get_T_in_dot(index) * DT
            if self.countercurrent and index == len(self.sections)-1:
                return
            self.sections[index].T_out += self.get_T_out_dot(index) * DT
    

class ExchangerSection:
    def __init__(self,dictionary,length):
        self.dictionary = dictionary
        self.length = length
        self.initialize_inside()
        self.initialize_outside()
    
    def initialize_inside(self):
        self.volume_inside = self.length* self.dictionary['D_int']**2 * np.pi / 4
        self.mass_inside = self.dictionary['rho_inside'] * self.volume_inside
        self.T_in = property_dictionary['t_ini_inside']
        self.w_in = property_dictionary['w_ini_inside']
        self.exchange_area = self.length * self.dictionary['D_int'] * np.pi
        
    def initialize_outside(self):
        self.volume_outside = (self.length*self.dictionary['D_ext']**2*np.pi / 4 ) - (self.length*self.dictionary['D_int']**2*np.pi / 4 )
        self.mass_outside = self.dictionary['rho_outside'] * self.volume_outside
        self.T_out = property_dictionary['t_ini_outside']
        self.w_out = property_dictionary['w_ini_outside']
        
    def set_T_in(self,T):
        self.T_in = T
        
    def set_T_out(self,T):
        self.T_out = T
        
    def set_w_in(self,w):
        self.w_in = w
        self.mass_inside = self.dictionary['rho_inside'] * self.volume_inside
    
    def set_w_out(self,w):
        self.w_out = w
        self.mass_outside = self.dictionary['rho_outside'] * self.volume_outside
    

    @property
    def get_q(self):
        return self.dictionary['U']*self.exchange_area*(self.T_in-self.T_out)


class Controller:
    
    def __init__(self,exchanger,temperatures):
        self.exchanger = exchanger
        self.temperatures = temperatures
    
    def set_boundary_temperatures(self, temperatures):
        self.temperatures = temperatures
    
    def fix_boundary_temperatures(self):
        self.exchanger.sections[0].set_T_in(self.temperatures[0])
        if self.exchanger.countercurrent:
            self.exchanger.sections[-1].set_T_out(self.temperatures[1])
            return
        self.exchanger.sections[0].set_T_out(self.temperatures[1])
    
    def set_boundary_outer_w(self,w):
        for i in range(len(self.exchanger.sections)):
            self.exchanger.sections[i].set_w_out(w) 
    
    def set_boundary_inner_w(self,w):
        for i in range(len(self.exchanger.sections)):
            self.exchanger.sections[i].set_w_in(w)       

        
            
class Sensor:
    def __init__(self, exchanger, v_desv, position = -1):
        self.storage = exchanger.storage
        self.position = position
        self.times = []
        self.inner_temperatures = []
        self.outer_initial_temperatures = []
        self.inner_initial_temperatures = []
        self.w_storage = []
        self.v_desv = v_desv
    
    def load_stationary_state(self):
        self.stationary_state = self.storage.stationary_state_inner_profile[-1][self.position]
        
    
    def read(self):
        inner_reading = self.storage.stored_final_inner_temperatures[-1]
        inner_initial_reading = self.storage.stored_initial_inner_temperatures[-1]
        if countercurrent:
            outer_reading = self.storage.stored_final_outer_temperatures[-1]
        else:
            outer_reading = self.storage.stored_initial_outer_temperatures[-1]
            
        flow = self.storage.stored_outer_wf[-1]
        time = self.storage.stored_times[-1]
        if self.v_desv:
            inner_reading = inner_reading - self.stationary_state
            inner_initial_reading = self.storage.stored_initial_inner_temperatures[-1] -exchanger.storage.stationary_state_inner_profile[0][0]
            if countercurrent:
                outer_reading = outer_reading - exchanger.storage.stationary_state_outer_profile[0][-1]
            else:
                outer_reading = outer_reading - exchanger.storage.stationary_state_outer_profile[0][0]
            flow = flow - Initial_Valve_State*Max_Valve_Flow
            
        noisy = self.noises(inner_reading,outer_reading,flow,inner_initial_reading)
        
        self.times.append(time)
        self.inner_temperatures.append(noisy[0])
        self.outer_initial_temperatures.append(noisy[1])
        self.inner_initial_temperatures.append(noisy[3])
        self.w_storage.append(noisy[2])
    
            
    def noises(self,inner_reading,outer_reading,flow_read,initial_inner_reading):
        inner = inner_reading + NOISES[0]*np.random.random_sample() - 0.5*NOISES[0]
        outer = outer_reading + NOISES[1]*np.random.random_sample() - 0.5*NOISES[1]
        flow = flow_read + NOISES[2]*np.random.random_sample() - 0.5*NOISES[2]
        inner_initial = initial_inner_reading + NOISES[2]*np.random.random_sample() - 0.5*NOISES[3]
        return inner,outer,flow,inner_initial
        
       

class Looper:
    def __init__(self,exchanger,valve):
        self.exchanger = exchanger
        self.valve = valve
        self.initial_time = 0
        self.initial_iteration = 0
        self.main_time = 0
        self.main_iteration = 0
        
    def initial_loop(self):
        while self.initial_time < tiempo_inicial_final:
            exchanger.integration()
            self.initial_time += DT
            self.initial_iteration +=1
            #if self.initial_iteration % 100 == 0:
            #    print(f'Iteration number {self.initial_iteration}')
            #    for i in exchanger.sections:
            #        print(i.T_in, i.T_out)
        exchanger.storage.create_stationary_state([exchanger.sections[i].T_in for i in range(exchanger.zones)],[exchanger.sections[i].T_out for i in range(exchanger.zones)])
        exchanger.sensor.load_stationary_state()
                    
    def main_loop(self,perturbation_interval,perturbation_range):
        time_between_perturbations = 0
        while self.main_time < tiempo_main_final:
            exchanger.integration()
            self.main_time += DT
            self.main_iteration +=1
            if time_between_perturbations > perturbation_interval:
                time_between_perturbations = 0
                change = 2*np.random.sample()*perturbation_range - perturbation_range
                exchanger.controller.set_boundary_temperatures([exchanger.controller.temperatures[0],exchanger.controller.temperatures[1]+change])
                
            if self.main_time > 70:
                valve.set_opening(0.8)
            exchanger.storage.add_times(self.main_time)
            exchanger.storage.add_wfs(exchanger.sections[0].w_in,exchanger.sections[0].w_out)
            exchanger.storage.add_temperatures(exchanger.sections[0].T_in,exchanger.sections[exchanger.zones-1].T_in,exchanger.sections[0].T_out,exchanger.sections[exchanger.zones-1].T_out)
            exchanger.storage.add_profiles([exchanger.sections[i].T_in for i in range(exchanger.zones)],[exchanger.sections[i].T_out for i in range(exchanger.zones)])   
            exchanger.sensor.read()    
            time_between_perturbations += DT     
            
    def training_loop(self,control_intervals,inner_temp_disturbance_data = [0,9999999,99999999,0],outer_temp_disturbance_data = [0,9999999,99999999,0]):#perturbation_base,perturbation_interval,perturbation_range,control_interval
        inner_disturbance_base ,inner_disturbance_interval_min, inner_disturbance_interval_max , inner_disturbance_range = inner_temp_disturbance_data[0] ,inner_temp_disturbance_data[1],inner_temp_disturbance_data[2] , inner_temp_disturbance_data[3] #perturbation_interval
        outer_disturbance_base ,outer_disturbance_interval_min, outer_disturbance_interval_max , outer_disturbance_range = outer_temp_disturbance_data[0] ,outer_temp_disturbance_data[1],outer_temp_disturbance_data[2] , outer_temp_disturbance_data[3]
        control_interval_min, control_interval_max = control_intervals[0], control_intervals[1]
        time_since_last_control_action = 0
        time_since_last_inner_disturbance = 0
        time_since_last_outer_disturbance = 0
        inner_disturbance_interval = inner_disturbance_interval_min + (inner_disturbance_interval_max - inner_disturbance_interval_min ) * np.random.random_sample()
        outer_disturbance_interval = outer_disturbance_interval_min + (outer_disturbance_interval_max - outer_disturbance_interval_min ) * np.random.random_sample()
        control_interval = control_interval_min + (control_interval_max - control_interval_min) * np.random.random_sample()
        while self.main_time < tiempo_main_final:
            exchanger.integration()
            self.main_time += DT
            time_since_last_inner_disturbance += DT
            time_since_last_outer_disturbance += DT              
            time_since_last_control_action += DT
            self.main_iteration +=1
            if time_since_last_inner_disturbance >= inner_disturbance_interval and T_ini_inside_changing:
                random = inner_disturbance_base + 2*np.random.random_sample()*inner_disturbance_range - inner_disturbance_range
                exchanger.controller.set_boundary_temperatures([random,exchanger.controller.temperatures[1]])
                time_since_last_inner_disturbance = 0
                inner_disturbance_interval = inner_disturbance_interval_min + (inner_disturbance_interval_max - inner_disturbance_interval_min ) * np.random.random_sample()
            
            if time_since_last_outer_disturbance >= outer_disturbance_interval and T_ini_outside_changing:
                random = outer_disturbance_base + 2*np.random.random_sample()*outer_disturbance_range - outer_disturbance_range
                exchanger.controller.set_boundary_temperatures([exchanger.controller.temperatures[0],random])
                time_since_last_outer_disturbance = 0
                outer_disturbance_interval = outer_disturbance_interval_min + (outer_disturbance_interval_max - outer_disturbance_interval_min ) * np.random.random_sample()
                
            if time_since_last_control_action >= control_interval:
                random = np.random.random_sample()
                valve.set_opening(random)
                time_since_last_control_action = 0
                control_interval = control_interval_min + (control_interval_max - control_interval_min) * np.random.random_sample()
                #if self.main_iteration % 100 == 0:
                #    print(f'Iteration number {self.main_iteration}')
                #    for i in exchanger.sections:
                #        print(i.T_in, i.T_out)
            exchanger.storage.add_times(self.main_time)
            exchanger.storage.add_wfs(exchanger.sections[0].w_in,exchanger.sections[0].w_out)
            exchanger.storage.add_temperatures(exchanger.sections[0].T_in,exchanger.sections[exchanger.zones-1].T_in,exchanger.sections[0].T_out,exchanger.sections[exchanger.zones-1].T_out)
            #exchanger.storage.add_profiles([exchanger.sections[i].T_in for i in range(exchanger.zones)],[exchanger.sections[i].T_out for i in range(exchanger.zones)])   
            exchanger.sensor.read()
            

            

class Storage:
    def __init__(self):
        self.stored_times = []
        self.stored_inner_wf = []
        self.stored_outer_wf = []
        self.stored_initial_inner_temperatures = []
        self.stored_final_inner_temperatures = []
        self.stored_initial_outer_temperatures = []
        self.stored_final_outer_temperatures = []
        self.stationary_state_inner_profile = []
        self.stationary_state_outer_profile = []
        self.stored_inner_profiles = []
        self.stored_outer_profiles = []
        
    def create_stationary_state(self,inner_profile,outer_profile):
        self.stationary_state_inner_profile.append(inner_profile)
        self.stationary_state_outer_profile.append(outer_profile)
        
    
    def add_profiles(self,inner_profile,outer_profile):
        self.stored_inner_profiles.append(inner_profile)
        self.stored_outer_profiles.append(outer_profile)
    
    def add_wfs(self,inner_wf,outer_wf):
        self.stored_inner_wf.append(inner_wf)
        self.stored_outer_wf.append(outer_wf)
    
    def add_times(self,time):
        self.stored_times.append(time)
        
    def add_temperatures(self,inner_initial,inner_final,outer_initial,outer_final):
        self.stored_initial_inner_temperatures.append(inner_initial)
        self.stored_final_inner_temperatures.append(inner_final)
        self.stored_initial_outer_temperatures.append(outer_initial)
        self.stored_final_outer_temperatures.append(outer_final)
    
    
    def plot_stored_temperatures(self , v_desv = False):
        if v_desv ==True:
            self.stored_initial_inner_temperatures = [self.stored_initial_inner_temperatures[i] - self.stationary_state_inner_profile[0][0] for i in range(len(self.stored_initial_inner_temperatures))]
            self.stored_final_inner_temperatures = [self.stored_final_inner_temperatures[i] - self.stationary_state_inner_profile[0][-1] for i in range(len(self.stored_initial_inner_temperatures))]
            self.stored_initial_outer_temperatures = [self.stored_initial_outer_temperatures[i] - self.stationary_state_outer_profile[0][0] for i in range(len(self.stored_initial_inner_temperatures))]
            self.stored_final_outer_temperatures = [self.stored_final_outer_temperatures[i] - self.stationary_state_outer_profile[0][-1] for i in range(len(self.stored_initial_inner_temperatures))]
        fig, temp_ax = plt.subplots()
        fig.set_size_inches(14, 8)
        inner_initial_lines = temp_ax.plot(self.stored_times, self.stored_initial_inner_temperatures, 'r')
        inner_final_lines = temp_ax.plot(self.stored_times, self.stored_final_inner_temperatures, 'y')
        outer_initial_lines = temp_ax.plot(self.stored_times, self.stored_initial_outer_temperatures, 'b')
        outer_final_lines = temp_ax.plot(self.stored_times, self.stored_final_outer_temperatures, 'c')
        meow = temp_ax.plot(self.stored_times,self.stored_outer_wf)
        plt.show()
    
    
    def plot_profiles(self,v_desv = False):
        if v_desv ==True:
                    self.stored_inner_profiles = [[self.stored_inner_profiles[j][i]-self.stationary_state_inner_profile[0][i] for i in range(len(self.stored_inner_profiles[j]))] for j in range(len(self.stored_inner_profiles))]
                    self.stored_outer_profiles = [[self.stored_outer_profiles[j][i]-self.stationary_state_outer_profile[0][i] for i in range(len(self.stored_outer_profiles[j]))]  for j in range(len(self.stored_inner_profiles))]
        self.fig, self.ax = plt.subplots()

        self.line2, = self.ax.plot(self.stored_inner_profiles[0], label='Outer Temperature')
        self.line1, = self.ax.plot(self.stored_outer_profiles[0], label='Inner Temperature')
        #dataset = np.array(self.stored_inner_profiles,self.stored_outer_profiles)
        ani = animation.FuncAnimation(self.fig, self.update, frames=1000, interval=50, blit=False)
        plt.show()
        
    def update(self,num):
 
        self.line1.set_ydata(self.stored_inner_profiles[num])
        self.line2.set_ydata(self.stored_outer_profiles[num])
        # Update the legend to show the current time
        self.ax.legend([self.line1, self.line2], [f'Inner Temperature (t={self.stored_times[num]:.1f} s)', f'Outer Temperature (t={self.stored_times[num]:.1f} s)'])
        self.ax.set_ybound(-5,5)
        return [self.line1, self.line2,self.ax]
    

class Valve:
    
    def __init__(self,exchanger,w_max,initial_opening):
        self.exchanger = exchanger
        self.w_max= w_max
        self.opening=initial_opening
        
    @property    
    def get_w(self):
        return self.w_max*self.opening
    
    def set_opening(self,opening):
        if opening < 0:
            self.opening = 0
            return
        if opening > 1:
            self.opening = 1
            return
        self.opening = opening
        self.apply()
    
    def apply(self):
        exchanger.controller.set_boundary_outer_w(self.get_w)



property_dictionary ={
            'U' : 1480,
            'cp_inside' : 4180,
            'cp_outside' : 3000,
            'D_ext' : 0.25,
            'D_int' : 0.15,
            'rho_outside' : 1000,
            'rho_inside' : 1000,
            't_ini_outside' : 20,
            't_ini_inside' : 40,
            'w_ini_outside': 10,
            'w_ini_inside': 7,
            'initial_t_entry_inside': 70,
            'initial_t_entry_outside': 25
            }
start = time.time()
tiempo = 0
DT = 0.25
tiempo_inicial_final = 1000
tiempo_main_final = 800000
ifila = 14
Nzonas = 100
length = 25
countercurrent = False 
v_desv = True
exchanger = Exchanger(length,property_dictionary,Nzonas,countercurrent,v_desv)
Initial_Valve_State = 0.5
Max_Valve_Flow = 20
valve = Valve(exchanger,Max_Valve_Flow,Initial_Valve_State)
looper = Looper(exchanger,valve)

T_ini_inside_changing = True
T_ini_outside_changing = True
SAVE_PROFILES = False
NOISES = [0.05,0.05,0.025,0.05]
def produce_training_data(): 
    looper.initial_loop()
    #looper.main_loop(60,2)
    looper.training_loop([195,735],[70,195,735,10],[25,195,735,5])
    #exchanger.storage.plot_stored_temperatures(True)
    training_inner_temperatures = exchanger.sensor.inner_temperatures
    training_initial_inner_temperatures = exchanger.sensor.inner_initial_temperatures
    training_timers = exchanger.sensor.times
    training_ws = exchanger.sensor.w_storage
    training_outer_temperatures = exchanger.sensor.outer_initial_temperatures


    array = np.array([training_outer_temperatures,training_initial_inner_temperatures,training_ws,training_inner_temperatures])
    array = np.transpose(array)
    #plt.plot(training_timers,training_outer_temperatures)
    df = pd.DataFrame(data = array, index = training_timers, columns = ['Outer Temp.','Initial Inner Temp.','Outer Ws' ,'Inner Temp.'])
    df = df.iloc[::4]
    if T_ini_inside_changing == False:
        df = df.drop(columns=['Initial Inner Temp.'])
    if T_ini_outside_changing ==False:
        df = df.drop(columns = ['Outer Temp.'])
    print(df)
    df.to_csv('simulated_data_sep_var_ambas_T_rn_150_2.csv',sep = ';')
    end = time.time()
    print(f'Elapsed time: {end-start} s')
    #plt.plot(training_timers,training_data)
    #exchanger.storage.plot_profiles(True)
    #plt.plot(exchanger.sensor.times,exchanger.sensor.internal_storage)
    #plt.show()

produce_training_data()