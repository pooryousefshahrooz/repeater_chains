#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time
import csv
import networkx as nx
import sys
import numpy as np
import math
from statistics import mean


# In[50]:


class Memory:
    def __init__(self,mode,repeater_id,entangled_node,memory_id,link_duration,
                 attached_link_duration,initial_fidelity,
                 EPR_gen_success_p,cut_off):
        self.mode = mode
        self.entangled_node = entangled_node
        self.entangled_memory_id = memory_id
        self.EPR_age = -1 # we track the age of the EPR pair generated for this memory
        self.memory_id = memory_id
        self.repeater_id = repeater_id
        self.generated_flag = False
        self.attempt_result = False
        self.expired = False
        self.attempt_flag = False# check if we have started the attempt 
        self.synchronous_flag = False # indicates you can not try even if you have failed untile next time slot
        self.attempt_start_time = 0 # we track when we have started the attempt
        self.attempt_finishing_time = 0 # when would we get the ack for the attempt
        self.initial_fidelity = initial_fidelity# track the fidleity of the EPR pair stored on this memory
        self.attached_link_duration = attached_link_duration # how long it takes to send photon one way
        self.attempt_success_p = EPR_gen_success_p
        self.cut_off = cut_off
        self.link_duration = link_duration
        self.each_time_slot_duration = 3*link_duration
        self.added_clock = 0
        self.memory_last_updated_clock = 0
        self.arrived_on_other_side_flag = False
        self.arrived_on_both_sides_flag = False
        
class Message:
    def __init__(self,experimenting_classical_communication_flag,repeater,destination,path_id,
                 message_type,left_node,right_node,memory_id_left,memory_id_right,
                 left_age,right_age,clock_counter,link_duration,success_flag):
        self.sender = repeater
        self.receiver = destination
        self.path_id = path_id
        self.start_sending_time = clock_counter
        if experimenting_classical_communication_flag:
            self.arriving_time = clock_counter+link_duration
        else:
            self.arriving_time = clock_counter+1
        self.memory_id_left= memory_id_left
        self.memory_id_right = memory_id_right
        self.left_node = left_node
        self.right_node = right_node
        self.left_qubit_age = left_age
        self.right_qubit_age = right_age
        self.type = message_type
        self.success_flag = success_flag
        self.processing_flag = False
        if global_printing_flag:
            print("repeater %s sends a %s message which will arrive at %s and now is %s"%(repeater,message_type,
                                                                                          self.arriving_time,
                                                                                          clock_counter))
            
class System:
    def __init__(self,mode,t_coh,R,path_id,end_time,q_value,cut_off,each_link_length,each_path_memory_min,each_path_memory_max):
        self.mode = mode
        self.R = R
        self.path_id= path_id
        self.end_time = end_time
        self.experimenting_classical_communication = experimenting_classical_communication
        self.having_cut_offs = having_cut_offs
        self.entanglement_generation_delay = entanglement_generation_delay
        self.considering_end_nodes_idle_time = considering_end_nodes_idle_time
        self.q_value = q_value
        self.e2e_EPRs = 0
        self.clock_counter = 0
        self.starting_time = 0
        self.last_delivery_time = 0
        self.all_delivery_durations = []
        self.last_generated = 0
        self.synchronous_flag = False
        self.all_generated_flag = False
        self.swap_operation_is_performed_flag = False
        self.initial_fidelity = initial_fidelity 
        self.cut_off= cut_off
        self.path_id_path_repeaters =path_id_path_repeaters
        self.path_id_path_links = path_id_path_links
        self.each_path_source_destination = each_path_source_destination
        self.each_link_cut_off  ={}
        self.each_link_length = each_link_length
        self.G = nx.Graph()
        for edge in self.path_id_path_links[self.path_id]:
            self.each_link_cut_off[edge] = self.cut_off
            self.G.add_edge(edge[0],edge[1],weight=self.each_link_length[edge])
        self.each_path_memory_min = each_path_memory_min
        self.each_path_memory_max = each_path_memory_max
        self.each_path_each_repeater_each_link_each_memory = {}
        self.each_path_repeater_left_right_memory_id_memory_object = {}
        self.each_repeater_left_right_memory_swap_flag = {}
        self.each_repeater_paired_nodes_memories = {}
        self.each_repeater_left_right_memory_exist_flag = {}
        self.each_repeater_left_right_memory_age_tracking_flag = {}
        self.each_path_swap_on_source_node_qubit_arrived={}
        self.each_path_swap_on_source_node_qubit_arrived={}
        self.each_path_swap_on_source_node_qubit_id={}
        self.each_path_swap_on_end_node_qubit_id = {}
        self.paralel_each_memory_attempt_flag ={}
        self.each_repeater_swap_flag = {}
        self.longest_link_duration = 0
        self.expired_qubits_counter = 0
        self.successfull_first_segment_geenration_times =[]
        self.message_channel = []
        self.swap_list = {}
        self.swap_start_sending_time_list = {}
        self.failed_at_least_one_link = False
        self.global_expiration_flag = False
        self.each_path_each_repeater_swaped_memory_ages ={}
        self.each_path_all_delivered_pairs_fidelity ={}
        self.each_path_all_delivered_pairs_fidelity_including_end_nodes = {}
        self.t_coh = t_coh
        self.F = {}
        self.each_repeater_left_right_memories_waiting_times = {}
        self.mu_i = {}
        self.mu = mu
        self.mu_i_value = mu_i_value
        self.last_generated_expiration_message = 0
        self.one_successful_round_times=[]
        self.last_delivered_e2e_epr_time = 0
        self.last_failed_time = 0
        self.last_be_geenrated_time = 0
        self.one_failed_round_times = []
        self.time_granularity_value = time_granularity_value
        self.each_path_swap_on_source_node_qubit_arrived[self.path_id]=False
        self.each_path_swap_on_source_node_qubit_id[self.path_id] = -1
        self.each_path_swap_on_end_node_qubit_id[self.path_id] = -1
        for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
            self.each_repeater_swap_flag[repeater] = False
            self.F[self.path_id,repeater]= 1
            try:
                self.mu_i[self.path_id].append(1)
            except:
                self.mu_i[self.path_id] = [1]
            self.swap_list[repeater]=[]
            self.swap_start_sending_time_list[repeater] = []   
        for repeater in self.path_id_path_repeaters[self.path_id][:-1]:
            for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
                self.paralel_each_memory_attempt_flag[self.path_id,repeater,memory_id] = False
        for repeater in self.path_id_path_repeaters[self.path_id]:
            for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
                if repeater == self.each_path_source_destination[self.path_id][0]:
                    self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"right",memory_id] =False
                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,repeater,"right",memory_id] =False
                elif repeater == self.each_path_source_destination[self.path_id][1]:
                    self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"left",memory_id] =False
                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,repeater,"left",memory_id] =False
                else:
                    self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"right",memory_id] =False
                    self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"left",memory_id] =False
                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,repeater,"right",memory_id] =False
                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,repeater,"left",memory_id] =False
        #if self.mode=="synch1" or self.mode=="synch2":
        link_lengths = []
        
        repeater_index = 0
        for link in self.path_id_path_links[self.path_id]:
            length = self.each_link_length[link]
            link_lengths.append(length)
            repeater_index+=1
        longest_link = max(link_lengths)
#         self.longest_link_duration = int((1.44*longest_link)/ 299792*1000000)
        self.longest_link_duration = int(longest_link/2e5*1000000)
        sum_link = sum(link_lengths)-link_lengths[len(link_lengths)-1]
        # sum_link = sum(link_lengths)
        self.e2e_communication_duration =int(sum_link/2e5*1000000)
#         for path_id,path_links in self.path_id_path_links.items():
        for link in self.path_id_path_links[self.path_id]:
            for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
                length = self.each_link_length[link]
 
                link_success_p = 10**(-0.2*length/10)
                # link_success_p = 1.0
                link_duration = int((1.44*length)/ 299792*1000000)
                link_duration = int(length/2e5*1000000)
                # link_duration = 2000
                left_memory_object_instanse = Memory(self.mode,link[0],link[1],memory_id,
                                                     link_duration,
                                       link_duration,self.initial_fidelity,
                                       link_success_p,self.cut_off)
                right_memory_object_instanse = Memory(self.mode,link[1],link[0],memory_id,link_duration,
                                       link_duration,self.initial_fidelity,
                                       link_success_p,self.cut_off)
                self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id] = left_memory_object_instanse
                self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[1],"left",memory_id] = right_memory_object_instanse


    
  
    
    def shortest_path_to_end_nodes_length(self,repeater):
        source = self.each_path_source_destination[self.path_id][0]
        destination = self.each_path_source_destination[self.path_id][1]
        shortest_path_length1  = nx.shortest_path_length(self.G, source=source, target=repeater,weight = "weight")
        shortest_path_length2  = nx.shortest_path_length(self.G, source=repeater, target=destination,weight = "weight")
        # if shortest_path_length2>shortest_path_length1:
        #     shortest_path_length1 = shortest_path_length2
        if global_printing_flag:
            print("Shortest path from %s to %s is %s which is %s microseconds"%(source,repeater,shortest_path_length1,int(shortest_path_length1/2e5*1000000)))
        return shortest_path_length1,source
    
        
                        
                        
    
                    


    
        
    def update(self,link,memory_id,current_clock_counter):
        expired_flag = False
#         print("going to attempt!")
        left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id]
        right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[1],"left",memory_id]
        left_node_memory_exist_flag = self.each_repeater_left_right_memory_exist_flag[self.path_id,link[0],"right",memory_id]
        right_node_memory_exist_flag = self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id]
#         if global_printing_flag:
#             print("attemptig generating EPR at %s on link (%s,%s) flags %s %s "%(current_clock_counter,
#                                                         left_memory_object.repeater_id,
#                                                         left_memory_object.entangled_node,
#                                                         left_node_memory_exist_flag,
#                                                         right_node_memory_exist_flag
#                                                                                                      ))

       
#         print("time %s flag on link %s,%s is %s memory exist %s %s generated flag %s"%(current_clock_counter,link[0],link[1],
#                                                    left_memory_object.attempt_flag,
#                                                   left_node_memory_exist_flag,
#                                                   right_node_memory_exist_flag,
#                                                   left_memory_object.generated_flag))
        if printing_qubit_aging_flag:
            print(" 0 time clock %s   link (%s,%s) last update (%s, %s) age track flag(%s,%s) got aged by one new age %s, %s FLAG %s "%(current_clock_counter,link[0],link[1],
                                                                                                                                   left_memory_object.memory_last_updated_clock,
                                                                                                                                   right_memory_object.memory_last_updated_clock,
                                                                                                                                   self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id],
                                                                                                                                   self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id],
                                                                                                                                   left_memory_object.EPR_age,right_memory_object.EPR_age,
                                                                                                                               left_memory_object.generated_flag))
        if  (not left_memory_object.attempt_flag and 
            not left_node_memory_exist_flag and 
            not right_node_memory_exist_flag and not left_memory_object.generated_flag):
            # not right_node_memory_exist_flag and not left_memory_object.generated_flag and self.none_expired()) :
#             if left_memory_object.synchronous_flag and left_memory_object.mode =="synch1":
#                 pass
#             else:
            left_memory_object.attempt_flag = True
            left_memory_object.entangled_node = link[1]
            right_memory_object.entangled_node = link[0]
            left_memory_object.attempt_start_time = current_clock_counter
            left_memory_object.EPR_age = -1
            right_memory_object.EPR_age = -1
            left_memory_object.expired = False
            right_memory_object.expired = False
            left_memory_object.arrived_on_other_side_flag = False
            left_memory_object.arrived_on_both_sides_flag = False
            right_memory_object.arrived_on_other_side_flag = False
            right_memory_object.arrived_on_both_sides_flag = False
            left_memory_object.qubit_storing_time = current_clock_counter
            self.each_repeater_left_right_memory_exist_flag[self.path_id,link[0],"right",memory_id] =False
            self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id] =False
            self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id]  = True
            self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id]  = False

            if self.experimenting_classical_communication:

                left_memory_object.attempt_finishing_time = current_clock_counter+2*left_memory_object.attached_link_duration+left_memory_object.added_clock
                left_memory_object.attempt_arriving_time_other_side=current_clock_counter+1*left_memory_object.attached_link_duration+left_memory_object.added_clock
            else:
                left_memory_object.attempt_finishing_time = current_clock_counter+1
            
                

            if printing_attempt_flag:
                print(" ******* statrt attemptig to generate EPR at %s on link (%s,%s) with p= %s that will finish at %s FLAG %s %s"%(current_clock_counter,
                                                                                                  left_memory_object.repeater_id,
                                                                                                  left_memory_object.entangled_node,left_memory_object.attempt_success_p,
                                                                                                 left_memory_object.attempt_finishing_time,
                                                                                                              left_memory_object.attempt_flag,
                                                                                                                          "\n"))
                
                
            left_memory_object.memory_last_updated_clock  = current_clock_counter
            right_memory_object.memory_last_updated_clock = current_clock_counter

                    
        elif left_memory_object.attempt_flag:#we have attempted before lets check if it is done

            if left_memory_object.memory_last_updated_clock<current_clock_counter or current_clock_counter==0:
                if self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id]:
                    left_memory_object.EPR_age = left_memory_object.EPR_age+self.time_granularity_value
                    left_memory_object.memory_last_updated_clock  = current_clock_counter
#                     print("1 we increased update time left_memory_object.EPR_age",left_memory_object.EPR_age)
            if right_memory_object.memory_last_updated_clock<current_clock_counter or current_clock_counter==0:
                if self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id]:
                    right_memory_object.EPR_age= right_memory_object.EPR_age+self.time_granularity_value
                    right_memory_object.memory_last_updated_clock = current_clock_counter
#                     print("12 we increased update time right_memory_object.EPR_age ",right_memory_object.EPR_age)
            EPR_age = left_memory_object.EPR_age
            right_EPR_age = right_memory_object.EPR_age
#             print("time %s we are comparing right age %s cut-off %s flag %s link_1 %s source_des %s"%(current_clock_counter,right_EPR_age ,self.each_link_cut_off[link] ,
#                                                                self.having_cut_offs , 
#                 link[1] , self.each_path_source_destination[self.path_id]))
            if (EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and 
                link[0] not in self.each_path_source_destination[self.path_id] and left_node_memory_exist_flag) or (
                right_EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and 
                link[1] not in self.each_path_source_destination[self.path_id] and right_node_memory_exist_flag):
                    expired_flag = True
                    self.expired_qubits_counter+=1
                    self.last_generated_expiration_message = current_clock_counter
                    if (EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and  
                        link[0] not in self.each_path_source_destination[self.path_id] and left_node_memory_exist_flag):
                        left_memory_object.expired = True
                        distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[0])
                    elif (right_EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and 
                          link[1] not in self.each_path_source_destination[self.path_id] and right_node_memory_exist_flag):
                        right_memory_object.expired = True
                        distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[1])
                    time_to_source = int(distance_from_source/2e5*1000000)
                    if not self.global_expiration_flag:
                        self.global_expiration_flag = True
                        self.expiration_message_arriving_time = current_clock_counter+time_to_source
                    if printing_qubit_expiration_flag:
                        print("0!!!!!!!!!!!!!!!!!!!!unfortunately the qubit got aged at link %s %s  age %s cut_off %s at time %s %s "%(left_memory_object.repeater_id,left_memory_object.entangled_node,left_memory_object.EPR_age, self.cut_off,current_clock_counter,
                                                                                                                                   "\n"))
                        import pdb
                        #pdb.set_trace()
                        if global_go_to_sleep_flag:
                            time.sleep(global_number_of_sleeping_sec_expiration)
    #                 self.send_a_message(link[0],self.path_id,"cancelling",left_memory_object.EPR_age,
    #                                                         right_memory_object.EPR_age,
    #                                                         link[0],
    #                                                         link[1],
    #                                                         left_memory_object.memory_id,
    #                                                         right_memory_object.memory_id,
    #                                                         current_clock_counter,False)

                    left_memory_object.attempt_flag = False
    #                 print("***************5************** we are setting the attempt flag for link %s,%s to False"%(link[0],link[1]))
                    
                    left_memory_object.EPR_age = -1
                    right_memory_object.EPR_age = -1

                    """we send a message to the source node to execute these commands locally there. 
                    We need to have a local version of this for links to know when they can attempt"""
                    self.each_repeater_left_right_memory_exist_flag[self.path_id,link[0],"right",memory_id] = False
                    self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id]  = False
                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id]  = False
                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id]  = False

        
        
            elif left_memory_object.attempt_arriving_time_other_side <= current_clock_counter and not left_memory_object.arrived_on_other_side_flag:
                random_value = random.uniform(0,1)
                left_memory_object.arrived_on_other_side_flag = True
                if random_value<= left_memory_object.attempt_success_p and left_memory_object.EPR_age+left_memory_object.attached_link_duration<self.each_link_cut_off[link]:
                    self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id] = True
                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id]  = True

                    photon_arrived_at_other_side = True
                    right_memory_object.EPR_age=-1
                    if printing_attempt_result_flag:
                        print("**** SUCCEFFULL ****** EPR pair on link %s,%s was arrived at other side at time %s! and it was successfule %s"%(left_memory_object.repeater_id,left_memory_object.entangled_node,current_clock_counter,
                                                                                                                                            "\n"))
                        if global_go_to_sleep_flag:
                            time.sleep(global_number_of_sleeping_sec_attempt_half_suc) 
                    left_memory_object.attempt_result= True
                    right_memory_object.generated_flag = True
                    right_memory_object.qubit_storing_time = current_clock_counter
                    self.last_generated = current_clock_counter+left_memory_object.attached_link_duration
                else:#
                    left_memory_object.attempt_result= False
                    photon_arrived_at_other_side = False
                    left_memory_object.EPR_age = -1
                    right_memory_object.EPR_age=-1

                    self.each_repeater_left_right_memory_exist_flag[self.path_id,link[0],"right",memory_id] = False
                    self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id]  = False
                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id]= False
                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id]= False
                    self.last_generated = current_clock_counter+left_memory_object.attached_link_duration
                    if printing_attempt_result_flag:
                        print("*****FAIL********  EPR pair on link %s,%s was arrived at other side at time %s! but it was NOT successfule %s"%(left_memory_object.repeater_id,left_memory_object.entangled_node,current_clock_counter,
                                                                                                                                            "\n"))
                        if global_go_to_sleep_flag:
                            time.sleep(global_number_of_sleeping_sec_attempt) 
                    left_memory_object.generated_flag = False
                    right_memory_object.generated_flag = False
            elif left_memory_object.attempt_finishing_time <= current_clock_counter and not left_memory_object.arrived_on_both_sides_flag:
                
                left_memory_object.arrived_on_both_sides_flag = True
                if left_memory_object.attempt_result and not left_memory_object.expired:
                    if printing_attempt_success_flag:
                        print("***FULLy *** EPR pair on link %s,%s was generated fully at time %s! (%s,%s) %s"%(left_memory_object.repeater_id,
                                                                                                                left_memory_object.entangled_node,
                                                                                                                current_clock_counter,
                                                                                                                2*left_memory_object.attached_link_duration,
                                                                                                                left_memory_object.attached_link_duration
                                                                                                                ,"\n"))
                        if global_go_to_sleep_flag:
                            time.sleep(global_number_of_sleeping_sec_attempt_full_suc)                        
                    #right_memory_object.qubit_storing_time = current_clock_counter
                    left_memory_object.attempt_flag = False
                    left_memory_object.generated_flag = True
                    left_memory_object.attempt_result = False
                    self.last_generated = current_clock_counter
#                     print("*****************1************ we are setting the attempt flag for link %s,%s to False"%(link[0],link[1]))

                   


                    if not self.entanglement_generation_delay:
                        left_memory_object.EPR_age = -1
                        right_memory_object.EPR_age = -1
                    left_memory_object.fidelity = self.initial_fidelity
                    self.each_repeater_left_right_memory_exist_flag[self.path_id,link[0],"right",memory_id] = True
                    self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id]  = True
                else:
                    if printing_attempt_result_flag:
                        print("*****fail**** EPR pair on link %s,%s was failed at time %s ! %s "%(left_memory_object.repeater_id,left_memory_object.entangled_node,current_clock_counter,"\n"))
                        if global_go_to_sleep_flag:
                            time.sleep(global_number_of_sleeping_sec_attempt_full_fail)  
                    self.last_generated = current_clock_counter
                    self.failed_at_least_one_link = True
                    left_memory_object.attempt_flag = False
                    left_memory_object.generated_flag = False
                    right_memory_object.generated_flag = False
#                     print("***************2************** we are setting the attempt flag for link %s,%s to False"%(link[0],link[1]))

                    left_memory_object.EPR_age = -1
                    right_memory_object.EPR_age = -1
#             else:
#                 print("for link %s no qubit has been expired with ages %s %s"%(link,left_memory_object.EPR_age,right_memory_object.EPR_age))

        elif not left_memory_object.attempt_flag and right_node_memory_exist_flag:#we may have attempted but done the swap and the qubit does not exist
            if left_memory_object.memory_last_updated_clock<current_clock_counter or current_clock_counter==0:
                if self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id]:
                    left_memory_object.EPR_age = left_memory_object.EPR_age+self.time_granularity_value
                    left_memory_object.memory_last_updated_clock  = current_clock_counter
#                     print("21 we increased update time")
            if right_memory_object.memory_last_updated_clock<current_clock_counter or current_clock_counter==0:
                if self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id]:
                    right_memory_object.EPR_age= right_memory_object.EPR_age+self.time_granularity_value
                    right_memory_object.memory_last_updated_clock = current_clock_counter
#                     print("22 we increased update time")
#             else:
#                 print("******************************at time %s flag for updating the age of qubits at link %s,%s is FALSE"%(current_clock_counter,link[0],link[1]))
            EPR_age = left_memory_object.EPR_age
            right_EPR_age = right_memory_object.EPR_age
            if (EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and 
                link[0] not in self.each_path_source_destination[self.path_id] and left_node_memory_exist_flag) or (
                right_EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and 
                link[1] not in self.each_path_source_destination[self.path_id] and right_node_memory_exist_flag):
                self.expired_qubits_counter+=1
                
                
                if printing_qubit_expiration_flag:
                    print("1***************1!!!!!!!!!!!!!!!!!!!!!! unfortunately the qubit got aged at link %s %s  age %s right_EPR_age %s cut_off %s at time %s %s"%(left_memory_object.repeater_id,left_memory_object.entangled_node,left_memory_object.EPR_age, right_EPR_age,self.cut_off,current_clock_counter,"\n"))
                    import pdb
                    if global_go_to_sleep_flag:
                        time.sleep(global_number_of_sleeping_sec_expiration)
                    #pdb.set_trace()
                # print("we expired becasue we are here")
                # time.sleep(10)
                expired_flag =True
                self.last_generated_expiration_message = current_clock_counter
                if (EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and  
                        link[0] not in self.each_path_source_destination[self.path_id] and left_node_memory_exist_flag):
                    left_memory_object.expired = True
                    distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[0])
                elif (right_EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and 
                      link[1] not in self.each_path_source_destination[self.path_id] and right_node_memory_exist_flag):
                    right_memory_object.expired = True
                    distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[1])
                time_to_source = int(distance_from_source/2e5*1000000)
                if not self.global_expiration_flag:
                    self.global_expiration_flag = True
                    if printing_qubit_expiration_flag:
                        print("we set the arriving time duration for expiration message to ",time_to_source)
                    self.expiration_message_arriving_time = current_clock_counter+time_to_source
            
#                 self.send_a_message(link[0],self.path_id,"cancelling",left_memory_object.EPR_age,
#                                                         right_memory_object.EPR_age,
#                                                         link[0],
#                                                         link[1],
#                                                         left_memory_object.memory_id,
#                                                         right_memory_object.memory_id,
#                                                         current_clock_counter,False)
                
                left_memory_object.attempt_flag = False
#                 print("****************3************* we are setting the attempt flag for link %s,%s to False"%(link[0],link[1]))

                left_memory_object.EPR_age = -1
                right_memory_object.EPR_age = -1
                self.each_repeater_left_right_memory_exist_flag[self.path_id,link[0],"right",memory_id] = False
                self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id]  = False
                self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id]  = False
                self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id]  = False

        
        else:
            if left_memory_object.memory_last_updated_clock<current_clock_counter or current_clock_counter==0:
                if self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id]:
                    left_memory_object.EPR_age = left_memory_object.EPR_age+self.time_granularity_value
                    left_memory_object.memory_last_updated_clock  = current_clock_counter
#                     print("21 we increased update time")
            if right_memory_object.memory_last_updated_clock<current_clock_counter or current_clock_counter==0:
                if self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id]:
                    right_memory_object.EPR_age= right_memory_object.EPR_age+self.time_granularity_value
                    right_memory_object.memory_last_updated_clock = current_clock_counter
#                     print("22 we increased update time")
#             else:
#                 print("******************************at time %s flag for updating the age of qubits at link %s,%s is FALSE"%(current_clock_counter,link[0],link[1]))
            EPR_age = left_memory_object.EPR_age
            right_EPR_age = right_memory_object.EPR_age
            if (EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and 
                link[0] not in self.each_path_source_destination[self.path_id] and left_node_memory_exist_flag) or (
                right_EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and 
                link[1] not in self.each_path_source_destination[self.path_id] and right_node_memory_exist_flag):
                self.expired_qubits_counter+=1
                
                
                if printing_qubit_expiration_flag:
                    print("1***************1!!!!!!!!!!!!!!!!!!!!!! unfortunately the qubit got aged at link %s %s  age %s right_EPR_age %s cut_off %s at time %s %s"%(left_memory_object.repeater_id,left_memory_object.entangled_node,left_memory_object.EPR_age, right_EPR_age,self.cut_off,current_clock_counter,"\n"))
                    import pdb
                    if global_go_to_sleep_flag:
                        time.sleep(global_number_of_sleeping_sec_expiration)
                    #pdb.set_trace()
                # print("we expired becasue we are here")
                # time.sleep(10)
                expired_flag =True
                self.last_generated_expiration_message = current_clock_counter
                if (EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and  
                        link[0] not in self.each_path_source_destination[self.path_id] and left_node_memory_exist_flag):
                    left_memory_object.expired = True
                    distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[0])
                elif (right_EPR_age >self.each_link_cut_off[link] and self.having_cut_offs and 
                      link[1] not in self.each_path_source_destination[self.path_id] and right_node_memory_exist_flag):
                    right_memory_object.expired = True
                    distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[1])
                time_to_source = int(distance_from_source/2e5*1000000)
                if not self.global_expiration_flag:
                    self.global_expiration_flag = True
                    self.expiration_message_arriving_time = current_clock_counter+time_to_source
            
#                 self.send_a_message(link[0],self.path_id,"cancelling",left_memory_object.EPR_age,
#                                                         right_memory_object.EPR_age,
#                                                         link[0],
#                                                         link[1],
#                                                         left_memory_object.memory_id,
#                                                         right_memory_object.memory_id,
#                                                         current_clock_counter,False)
                
                left_memory_object.attempt_flag = False
#                 print("****************3************* we are setting the attempt flag for link %s,%s to False"%(link[0],link[1]))

                left_memory_object.EPR_age = -1
                right_memory_object.EPR_age = -1
                self.each_repeater_left_right_memory_exist_flag[self.path_id,link[0],"right",memory_id] = False
                self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id]  = False
                self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id]  = False
                self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id]  = False

#             else:
#                 print("left_memory_object.attempt_flag off for link %s no qubit has been expired with ages %s %s"%(link,left_memory_object.EPR_age,right_memory_object.EPR_age))

        
#         if printing_qubit_aging_flag:
#             print(" 0 time clock %s   link (%s,%s) last update (%s, %s) age track flag(%s,%s) got aged by one new age %s, %s "%(current_clock_counter,link[0],link[1],
#                                                                                                                                    left_memory_object.memory_last_updated_clock,
#                                                                                                                                    right_memory_object.memory_last_updated_clock,
#                                                                                                                                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id],
#                                                                                                                                    self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id],
#                                                                                                                                    left_memory_object.EPR_age,right_memory_object.EPR_age))
        
        
        
        
        
        
        
        if left_memory_object.memory_last_updated_clock<current_clock_counter or current_clock_counter==0:
            if self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id]:
                left_memory_object.EPR_age = left_memory_object.EPR_age+self.time_granularity_value
                # these two commands indicates we have updates these two memories at this time
                left_memory_object.memory_last_updated_clock  = current_clock_counter
#                 print("31 we increased update time")
        if right_memory_object.memory_last_updated_clock<current_clock_counter or current_clock_counter==0:
            if self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id]:
                right_memory_object.EPR_age= right_memory_object.EPR_age+self.time_granularity_value
                # these two commands indicates we have updates these two memories at this time
                right_memory_object.memory_last_updated_clock = current_clock_counter
#                 print("32 we increased update time")
        # if printing_qubit_aging_flag:
        #     print(" 1 time clock %s   link (%s,%s) last update (%s, %s) age track flag(%s,%s) got aged by one new age %s, %s FLAG %s "%(current_clock_counter,link[0],link[1],
        #                                                                                                                            left_memory_object.memory_last_updated_clock,
        #                                                                                                                            right_memory_object.memory_last_updated_clock,
        #                                                                                                                            self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[0],"right",memory_id],
        #                                                                                                                            self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,link[1],"left",memory_id],
        #                                                                                                                            left_memory_object.EPR_age,right_memory_object.EPR_age,
        #                                                                                                                        left_memory_object.generated_flag))
        left_memory_object.memory_last_updated_clock  = current_clock_counter
        right_memory_object.memory_last_updated_clock = current_clock_counter
        
        
        
                                                                                                                
                  
        
    
    
    def send_a_message(self,repeater,message_type,left_node_qubit_age,
                       right_node_qubit_age,left_node,right_node,
                       memory_id_left,memory_id_right,
                       current_clock_counter,success_flag):
        length,destination = self.shortest_path_to_end_nodes_length(repeater)
#         link_duration = int((1.44*length)/ 299792*1000000)
        link_duration = int(length/2e5*1000000)
        # link_duration = 2000
        right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,repeater,"right",memory_id_right]

        if abs(current_clock_counter-right_memory_object.attempt_start_time-right_node_qubit_age)>2*self.time_granularity_value:
            print("it seems these two do not match right memory age %s and the reported age %s "%(current_clock_counter-right_memory_object.attempt_start_time ,right_node_qubit_age))
            import pdb
            pdb.set_trace()
        message = Message(self.experimenting_classical_communication,repeater,destination,self.path_id,
                          message_type,left_node,right_node,memory_id_left,memory_id_right,left_node_qubit_age,
                          right_node_qubit_age,current_clock_counter,link_duration,success_flag)
        self.message_channel.append(message)
    def receive_a_fizzling_message(self,message,clock_counter):
        global global_printing_flag
        if global_printing_flag:
            print("we received a fizzling message from %s for the starting time %s arrived at %s "%(message.sender,
                                                                                                message.start_sending_time,clock_counter))
        for repeater,start_sending_times in self.swap_start_sending_time_list.items():
            remaining_swap_start_sending_time = []
            for swap_start_sending_time in start_sending_times:
                if swap_start_sending_time<message.start_sending_time:
                    if len(self.swap_list[repeater])==1:
                        self.swap_list[repeater] =[]
                    elif len(self.swap_list[repeater])>1:
                        self.swap_list[repeater] = self.swap_list[repeater][1:]
                else:
                    remaining_swap_start_sending_time.append(swap_start_sending_time)
            self.swap_start_sending_time_list[repeater] = remaining_swap_start_sending_time
    def receive_a_swap_result_message(self,message,clock_counter):
        global global_printing_flag
        self.swap_list[message.sender].append(1)
        self.swap_start_sending_time_list[message.sender].append(message.start_sending_time)
        #if global_swap_tracking_flag:
            #print("swap result arrived from %s at %s on qubits with ages %s %s !rec./unrec. swap results: %s "%(message.sender,clock_counter,message.left_qubit_age,message.right_qubit_age,self.swap_list))
        if message.left_node==self.each_path_source_destination[message.path_id][0]:
#             print("************************************ this arrived swap is entangeled with the source node")
            self.each_path_swap_on_source_node_qubit_arrived[message.path_id] = True
            self.each_path_swap_on_source_node_qubit_id[message.path_id] = message.memory_id_right
        if message.right_node==self.each_path_source_destination[message.path_id][1]:
            self.each_path_swap_on_end_node_qubit_id[message.path_id]=message.memory_id_right

    def check_swap_results_arrived(self,clock_counter):
        messages_to_be_removed= []
        for message in self.message_channel:
            if message.arriving_time<=clock_counter and not message.processing_flag:
                message.processing_flag = True
                if message.type =="cancelling":
                    self.receive_a_fizzling_message(message,clock_counter)
                elif message.type =="swap_result":
                    self.receive_a_swap_result_message(message,clock_counter)
                    messages_to_be_removed.append(message)
                    
        for message in messages_to_be_removed:
            left_age=message.left_qubit_age
            right_age = message.right_qubit_age
            repeater = message.sender
            if global_swap_tracking_flag:
                print("swap results arrived from %s left memory elapsed time %s right memory elapsed time %s at time %s "%(repeater,left_age,right_age,clock_counter))
                # time.sleep(global_number_of_sleeping_sec_swap)                
            try:
                self.each_path_each_repeater_swaped_memory_ages[self.path_id][repeater]=[(left_age,right_age)]
            except:
                try:
                    self.each_path_each_repeater_swaped_memory_ages[self.path_id][repeater] = [(left_age,right_age)]
                except:
                    self.each_path_each_repeater_swaped_memory_ages[self.path_id]={}
                    self.each_path_each_repeater_swaped_memory_ages[self.path_id][repeater] = [(left_age,right_age)]
            self.message_channel.remove(message)
        if self.each_path_swap_on_source_node_qubit_arrived[self.path_id]:
            missing_one_swap = False
            for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
                if len(self.swap_list[repeater])>0:
                    if self.swap_list[repeater][0]==0:
                        missing_one_swap= True
                else:
                    missing_one_swap= True
            if not missing_one_swap:
#                 print("**************************************************** we release the source node qubit at time %s by settign it to False "%(clock_counter))
                self.each_repeater_left_right_memory_exist_flag[self.path_id,self.each_path_source_destination[path_id][0],"right",self.each_path_swap_on_source_node_qubit_id[path_id]] =False
                self.each_repeater_left_right_memory_exist_flag[self.path_id,self.each_path_source_destination[path_id][1],"left",self.each_path_swap_on_end_node_qubit_id[path_id]] =False
                self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,self.each_path_source_destination[path_id][0],"right",self.each_path_swap_on_source_node_qubit_id[path_id]] =False
                self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,self.each_path_source_destination[path_id][1],"left",self.each_path_swap_on_end_node_qubit_id[path_id]] =False
                
            
    def check_all_swaps(self,clock_counter):
        missing_one_swap = False
        for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
            if len(self.swap_list[repeater])>0:
                if self.swap_list[repeater][0]==0:
                    missing_one_swap= True
            else:
                missing_one_swap= True
                
        if not missing_one_swap:
            if global_delivered_EPR_flag:
                print("********************* we delivered one e2e EPR pair at time %s *********messages in channel %s***************"%(clock_counter,len(self.message_channel)))
            
            each_repeater_left_right_memory_idle_time = {}
            for path,repeater_times in self.each_path_each_repeater_swaped_memory_ages.items():
                for repeater,w_time in repeater_times.items():
                    new_waiting_time = w_time
                    each_repeater_left_right_memory_idle_time[repeater] = new_waiting_time

            self.each_path_each_repeater_swaped_memory_ages ={}
            f_e2e,e2e_fidelity=self.compute_f_e2e(each_repeater_left_right_memory_idle_time)
            try:
                self.each_path_all_delivered_pairs_fidelity[self.path_id].append(f_e2e)
            except:
                self.each_path_all_delivered_pairs_fidelity[self.path_id] = [f_e2e]
            try:
                self.each_path_all_delivered_pairs_fidelity_including_end_nodes[self.path_id].append(e2e_fidelity)
            except:
                self.each_path_all_delivered_pairs_fidelity_including_end_nodes[self.path_id] = [e2e_fidelity]
            
            
            for link in self.path_id_path_links[self.path_id]:
                for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
                    left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id]
                    right_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[1],"left",memory_id]
                    if left_memory_object.attempt_flag:
                        print("we should not be attempting on link %s,%s before receiving the swap results!"%(link[0],link[1]))
                        import pdb
                        pdb.set_trace()
                    left_memory_object.generated_flag = False
                    right_memory_object.generated_flag = False
                    left_memory_object.expired = False
                    self.each_path_swap_on_source_node_qubit_arrived[self.path_id]=False
                    self.each_path_swap_on_source_node_qubit_id[self.path_id] = -1
                    self.each_path_swap_on_end_node_qubit_id[self.path_id] = -1

                    self.make_all_memories_free()
                    self.all_generated_flag = False
            for repeater in self.path_id_path_repeaters[self.path_id][1:-1]: 
                if len(self.swap_list[repeater])==1:
                    self.swap_list[repeater] =[]
                    self.swap_start_sending_time_list[repeater] = []
                elif len(self.swap_list[repeater])>1:
                    self.swap_list[repeater] = self.swap_list[repeater][1:]
                    self.swap_start_sending_time_list[repeater] = self.swap_start_sending_time_list[repeater][1:]
            self.each_path_swap_on_source_node_qubit_arrived[self.path_id]=False
            self.each_path_swap_on_source_node_qubit_id[self.path_id] = -1
            self.each_path_swap_on_end_node_qubit_id[self.path_id] = -1
            
            
            
            for link in self.path_id_path_links[self.path_id]:
                for memory_id in range(self.each_path_memory_min[path_id],self.each_path_memory_max[self.path_id]):
                    left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id]
                    left_memory_object.generated_flag = False
                    left_memory_object.expired = False



            self.one_successful_round_times.append(clock_counter - self.last_delivered_e2e_epr_time)
            
            self.last_delivered_e2e_epr_time = clock_counter
            self.last_failed_time = clock_counter
            self.global_expiration_flag = False
            self.make_all_memories_free()
            import time
            if global_delivered_EPR_flag:
                if global_go_to_sleep_flag:
                    time.sleep(global_number_of_sleeping_sec_delivered)
            self.e2e_EPRs+=1
            self.swap_operation_is_performed_flag = False
            self.round_flag = True
            return True
        else:
            return False
    
    def swap_operation(self,current_clock_counter,repeater):
        global global_printing_flag
        left_side_memory_ages = []
        right_side_memory_ages = []
        left_memory_objects = []
        right_memory_objects = []
        for memory_id in range(self.each_path_memory_min[path_id],self.each_path_memory_max[self.path_id]):
            left_side_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,repeater,"left",memory_id]
            right_side_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,repeater,"right",memory_id]
            node_left_memory_id_exist_flag = self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"left",memory_id]
            node_right_memory_id_exist_flag = self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"right",memory_id]
            if node_left_memory_id_exist_flag:
                if left_side_memory_object.EPR_age not in left_side_memory_ages:
                    left_side_memory_ages.append(left_side_memory_object.EPR_age)
                    left_memory_objects.append(left_side_memory_object)
            if node_right_memory_id_exist_flag:
                if right_side_memory_object.EPR_age not in right_side_memory_ages:
                    right_side_memory_ages.append(right_side_memory_object.EPR_age)
                    right_memory_objects.append(right_side_memory_object)
        left_side_memory_ages = list(set(left_side_memory_ages))
        right_side_memory_ages = list(set(right_side_memory_ages))
        left_side_memory_ages.sort(reverse=True)
        right_side_memory_ages.sort(reverse=True)
        counter = 0
        for old_left in left_side_memory_ages:
            swapped_with_one_EPR = False
            for left_memory_object in left_memory_objects:
                if left_memory_object.EPR_age ==old_left:
                    for old_right in right_side_memory_ages:
                        for right_memory_object in right_memory_objects:
                            if right_memory_object.EPR_age==old_right:
                                if (left_memory_object.EPR_age >self.cut_off or 
                                    right_memory_object.EPR_age > self.cut_off or
                                    counter>=1 or not left_memory_object.generated_flag or not right_memory_object.generated_flag):
                                    import pdb
                                    print("this is a flaw in the code. we are not supposed to do swap on expired qubits!",left_memory_object.generated_flag,right_memory_object.generated_flag,left_memory_object.EPR_age,right_memory_object.EPR_age,counter)
                                    pdb.set_trace()
                                
                                counter+=1
                                swapped_with_one_EPR =True
                                random_value  = random.uniform(0,1)
#                                 print("turning the memory flag on right of %s to False as we did swap at %s "%(repeater,repeater))
#                                 print("turning the memory flag on left of %s to False as we did swap at %s "%(repeater,repeater))
                                self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"right",right_memory_object.memory_id] =False
                            
                                self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"left",left_memory_object.memory_id] =False
            
                                self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,repeater,"right",right_memory_object.memory_id] =False                            
                                self.each_repeater_left_right_memory_age_tracking_flag[self.path_id,repeater,"left",left_memory_object.memory_id] =False
            
                                self.failed_at_least_one_link = False
                                self.each_repeater_swap_flag[repeater]= True
                                left_entangled_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,left_memory_object.entangled_node,"right",left_memory_object.entangled_memory_id]
                                right_entangled_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,right_memory_object.entangled_node,"left",right_memory_object.entangled_memory_id]

                                
                                self.send_a_message(repeater,"swap_result",left_memory_object.EPR_age,
                                                    right_memory_object.EPR_age,
                                                    left_memory_object.entangled_node,
                                                    right_memory_object.entangled_node,
                                                    left_memory_object.memory_id,
                                                    right_memory_object.memory_id,
                                                    current_clock_counter,True)
                                if global_swap_tracking_flag:
                                    print("*************************************************** we performed swap at repeater %s at time %s on ages left %s right %s and we ce of qubits to FALSE "%(repeater,
                                                                                                                                                current_clock_counter,
                                                                                                                                                left_memory_object.EPR_age,
                                                                                                                                               right_memory_object.EPR_age
                                                                                                                                               ))
                                    # time.sleep(global_number_of_sleeping_sec_swap)
                                    
                                self.swap_operation_is_performed_flag = True
                                left_memory_object.attempt_flag = False
                                right_memory_object.attempt_flag = False
                                left_memory_object.EPR_age = -1
                                right_memory_object.EPR_age = -1
                                    


    def check_swap_operation_is_performed_on_all(self):
        for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
            if not self.each_repeater_swap_flag[repeater]:
                return False
        return True

    
    def all_are_generated(self):
        for repeater in self.path_id_path_repeaters[self.path_id]:
            for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
                if repeater != self.each_path_source_destination[self.path_id][0]:
                    left_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,repeater,"left",memory_id]
                    if not left_memory_object.generated_flag:
                        return False
                        
        self.all_generated_flag = True
        self.round_flag = False# this means all links have generated a link. even if they are expired 
        if global_printing_flag:
            print("***************************we have generated on all links******************")
        return True
      


                
#                 try:
#                     node_left_memory_id_generated_flag = self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"left",memory_id]
#                     node_right_memory_id_generated_flag = self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"right",memory_id]
#                     right_memory_id_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,repeater,"right",memory_id]
#                     left_memory_id_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,repeater,"left",memory_id]
#                     node_right_memory_id_generated_flag = right_memory_id_memory_object.generated_flag
#                     node_left_memory_id_generated_flag = left_memory_id_memory_object.generated_flag
#                 except:
#                     if global_printing_flag:
#                         print("**************0*************we have not generated on all links******************")
#                     return False
# #                 if not node_left_memory_id_generated_flag:
#                 if not node_right_memory_id_generated_flag or not node_left_memory_id_generated_flag:
#                     # if global_printing_flag:
#                     #     print("***************************we have not generated on all links******************")
#                     return False
                

#         self.all_generated_flag = True
#         self.round_flag = False# this means all links have generated a link. even if they are expired 
#         return True
    def none_expired(self):
        source= self.each_path_source_destination[self.path_id][0]
        end = self.each_path_source_destination[self.path_id][1]
        for link in self.path_id_path_links[self.path_id]:
            for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
                left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id]
                right_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[1],"left",memory_id]

                if link[0] == source:
                    if right_memory_object.expired:
                        if global_printing_flag:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! the right qubit at link %s,%s is expired!!!!"%(link[0],link[1]))
                        return False
                elif link[1] == end:
                    if left_memory_object.expired:
                        if global_printing_flag:
                            print("!!!!!!!!!!!!!!!!!!!!1!!!!!!!!!!!!!!!!! the left qubit at link %s,%s is expired!!!!"%(link[0],link[1]))
                        return False
                else:
                    if left_memory_object.expired or right_memory_object.expired:
                        if global_printing_flag:
                            print("!!!!!!!!!!!!!!!!!!2!!!!!!!!!!!!!!!!!!! the left qubit at link %s,%s is expired!!!!"%(link[0],link[1]))
                        return False
        # if global_printing_flag:
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! none of the qubits is expired!!!!")
        return True
    
    def make_all_memories_free(self):   

        source= self.each_path_source_destination[self.path_id][0]
        end = self.each_path_source_destination[self.path_id][1]
        for link in self.path_id_path_links[self.path_id]:
            for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
                left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id]
                right_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[1],"left",memory_id]
                left_memory_object.attempt_flag = False
                right_memory_object.attempt_flag = False    
                right_memory_object.generated_flag = False
                left_memory_object.generated_flag = False
                left_memory_object.expired = False
                right_memory_object.expired = False
                left_memory_object.EPR_age = -1
                right_memory_object.EPR_age = -1
                self.each_repeater_left_right_memory_exist_flag[self.path_id,link[0],"right",memory_id] =False
                self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id] =False

        
        messages_to_be_removed= []
        for message in self.message_channel:
            message.processing_flag = True
            messages_to_be_removed.append(message)
                    
        for message in messages_to_be_removed:
            left_age=message.left_qubit_age
            right_age = message.right_qubit_age
            repeater = message.sender
            # try:
            #     self.each_path_each_repeater_swaped_memory_ages[self.path_id,repeater].append((left_age,right_age))
            # except:
            #     self.each_path_each_repeater_swaped_memory_ages[self.path_id,repeater] = [(left_age,right_age)]
            self.message_channel.remove(message)

        for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
            self.swap_list[repeater]=[]

                
        self.global_expiration_flag = False
        if global_printing_flag:
            print("***************************we maid all memories free *****************")
            
        
    def h(self,e_x):
        if e_x<1e-9 or (1-e_x)<1e-9:
            return 0
        else:
            return -e_x*np.log2(e_x)-(1-e_x)*np.log2(1-e_x)
    def compute_f_e2e(self,each_repeater_left_right_memory_idle_time):
        sum_of_time = 0
        for repeater,left_right_time in each_repeater_left_right_memory_idle_time.items():
            if global_printing_flag:
                if global_delivery_printing_flag:
                    print("swap on repeater %s happened on qubits with age %s and %s have %s Rs "%(repeater,
                                                                                                left_right_time[0],
                                                                                                left_right_time[1],
                                                                                                len(list(each_repeater_left_right_memory_idle_time.keys()))))
            sum_of_time = sum_of_time+left_right_time[0][0]/1000000+left_right_time[0][1]/1000000#(4,10)
           
        source = self.each_path_source_destination[self.path_id][0]
        destination = self.each_path_source_destination[self.path_id][1]
        
        source_node_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,source,"right",0]
        destination_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,destination,"left",0]
        
    


        
        
        if global_delivery_printing_flag:
            print("end node source destination idle time %s destination idle time %s "%(source_node_memory_object.EPR_age,
                                                                                        
                                                                                       destination_node_memory_object.EPR_age,
                                                                                        
                                                                                           
                                                                                      ))
        


        sum_of_time_for_fidelity = sum_of_time+source_node_memory_object.EPR_age/1000000+destination_node_memory_object.EPR_age/1000000
        # print("our self.t_coh %s our sum_of_time_for_fidelity %s "%(self.t_coh,sum_of_time_for_fidelity))
        t_all = sum_of_time_for_fidelity/self.t_coh
        e2e_fidelity=1/2+1/2*(np.exp(-t_all))
        
        t = (sum_of_time)/self.t_coh
        exp = np.exp(-t)
        f_e2e=1/2+1/2*(exp)
        f_e2e=1/2+1/2*(np.exp(-t))
        return f_e2e,e2e_fidelity
        
      
    
    
    def sekret_key(self,path_id,n,rate):
        avg_e2e_f = sum(self.each_path_all_delivered_pairs_fidelity[path_id])/len(self.each_path_all_delivered_pairs_fidelity[path_id])
        avg_e2e_f_including_end_nodes = sum(self.each_path_all_delivered_pairs_fidelity_including_end_nodes[path_id])/len(self.each_path_all_delivered_pairs_fidelity_including_end_nodes[path_id])

        
        mu_e2e=(self.mu**n)*np.prod(self.mu_i[self.path_id])
        
        e_z =(1+mu_e2e)/2-(mu_e2e*avg_e2e_f)
        e_x=(1-mu_e2e)/2
        r= 1-self.h(e_x)-self.h(e_z)
        S = rate*r

        # def h(p_list):
        #     y_list = np.zeros(len(p_list))
        #     for i,p in enumerate(p_list):
        #         if p<1e-6 or (1-p)<1e-6:
        #             y_list[i]= 0
        #         else:
        #             y_list[i]= -p*np.log2(p)-(1-p)*np.log2(1-p)
        #     return y_list
        # N_links = 4
        # mu_link = 1
        # F_link = 1
        # mu_e2e = mu_link**(2*N_links-1)
        # # secret key rate calculations
        # ex = (1 - mu_e2e)/2
        # ez = (1 + mu_e2e)/2 - mu_e2e * avg_e2e_f
        # S = rate * (1-h([ex])-h([ez]))

        
        return S,r,mu_e2e,avg_e2e_f,e_x,e_z,avg_e2e_f_including_end_nodes
    def main(self):
        self.e2e_EPRs = 0
        clock_counter = 1
        while(clock_counter <=self.end_time):
            if clock_counter%5000==0:
                print("mode %s cut_off %s link %s R %s clock %s End %s "%(self.mode,self.cut_off,
                                                                          self.each_link_length[(0,1)],
                                                                          self.R,clock_counter,self.end_time),end="\r")
#             if not self.round_flag: 
#                 """this means one round of the paralel scheme has not finished. 
#                 We wait until we receive the swap results from all repeaters"""
#             self.check_swap_results_arrived(clock_counter)
#             self.check_all_swaps(clock_counter)
            
            for path_id,path_links in self.path_id_path_links.items():
                for link in path_links:
                    for memory_id in range(self.each_path_memory_min[path_id],self.each_path_memory_max[self.path_id]):
                        self.update(link,memory_id,clock_counter) 
                for link in path_links:
                    for memory_id in range(self.each_path_memory_min[path_id],self.each_path_memory_max[self.path_id]):
                        for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
                            self.swap_operation(clock_counter,repeater)  
                self.check_swap_results_arrived(clock_counter)
                if self.check_all_swaps(clock_counter):
                    clock_counter-=1
                if not self.none_expired():
                    # if max(self.last_generated_expiration_message+self.e2e_communication_duration,self.last_be_geenrated_time+self.e2e_communication_duration) <= clock_counter:
                    # if self.last_generated_expiration_message+self.e2e_communication_duration <= clock_counter:
                    if self.expiration_message_arriving_time<=clock_counter:
                        
                        if global_go_to_sleep_flag:
                            print("lets free all memories and start a new round")
                            time.sleep(2)
                        for link in self.path_id_path_links[self.path_id]:
                            for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
                                left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id]
                                left_memory_object.generated_flag = False
                                left_memory_object.expired = False
                                self.each_path_swap_on_source_node_qubit_arrived[self.path_id]=False
                                self.each_path_swap_on_source_node_qubit_id[self.path_id] = -1
                                self.each_path_swap_on_end_node_qubit_id[self.path_id] = -1
                                self.make_all_memories_free()
                                self.all_generated_flag = False
                                self.swap_operation_is_performed_flag = False
                                self.last_generated_expiration_message = clock_counter
                
                        self.one_failed_round_times.append(clock_counter-self.last_failed_time)
                        # print("*******************************we failed and it took %s for us to fail at time %s "%(clock_counter-self.last_failed_time,clock_counter))
                        # time.sleep(5)
                        self.last_failed_time = clock_counter
                        self.last_delivered_e2e_epr_time = clock_counter
                        self.last_generated_expiration_message = clock_counter
                        clock_counter-=1
            clock_counter+=self.time_granularity_value






                


# In[ ]:


def rand_repeater_chain(distance,Nrepeater,min_dist = 2):
    repeater_loc = np.concatenate(([0],np.sort(np.random.rand(Nrepeater)),[1]))*distance
    while np.min(np.diff(repeater_loc))< min_dist:
        repeater_loc = np.concatenate(([0],np.sort(np.random.rand(Nrepeater)),[1]))*distance    
    return repeater_loc
    # distance = 100 # in km
    # Nrepeater = 6
    
    
def compute_edges(experiment,distance,Nrepeater):
    each_edge_length = {}
    if experiment=="random_placement":
        left_link_length = 0
        each_link_length = {(0,1):left_link_length,(1,2):left_link_length,(2,3):left_link_length,
                    (3,4):left_link_length,
                   (4,5):left_link_length,
                   (5,6):left_link_length,
                    (6,7):left_link_length,
                   (7,8):left_link_length,
                   (8,9):left_link_length,
                   (9,10):left_link_length,
                   (10,11):left_link_length,
                   (11,12):left_link_length}

        locs = rand_repeater_chain(distance,Nrepeater)
        locs = np.array(locs)
        print(locs)
        point = locs[0]
        repeater_indx = 0
        for place in locs[1:]:
            print("length is ",place-point)
            each_link_length[(repeater_indx,repeater_indx+1)] = place-point
            repeater_indx +=1
            point  = place
        return each_link_length,[distance]
    elif experiment =="equal":
        new_n = (2+(Nrepeater-1))
        left_link_length = np.array([distance/new_n]*new_n)
        left_link_length = left_link_length[0]

        each_link_length = {(0,1):left_link_length,(1,2):left_link_length,(2,3):left_link_length,
                    (3,4):left_link_length,
                   (4,5):left_link_length,
                   (5,6):left_link_length,
                    (6,7):left_link_length,
                   (7,8):left_link_length,
                   (8,9):left_link_length,
                   (9,10):left_link_length,
                   (10,11):left_link_length,
                   (11,12):left_link_length}

        return each_link_length,[distance]
    elif experiment =="repeater_position":
        δ = 0.1
        rep_loc = np.linspace(δ,1-δ,20)
        Le2e = distance
        for i, pos in enumerate(rep_loc):
            L1 = pos*Le2e
            L2 = Le2e - L1
            L0 = pos
            left_link_length = L1
        
            each_link_length = {(0,1):left_link_length,(1,2):L2,(2,3):left_link_length,
                    (3,4):left_link_length,
                   (4,5):left_link_length,
                   (5,6):left_link_length,
                    (6,7):left_link_length,
                   (7,8):left_link_length,
                   (8,9):left_link_length,
                   (9,10):left_link_length,
                   (10,11):left_link_length,
                   (11,12):left_link_length}
        return each_link_length,rep_loc


# In[1]:


global_printing_flag = False
printing_qubit_update_flag = False
end_nodes_finit_coherence_time = False
printing_qubit_expiration_flag = False
printing_qubit_aging_flag = False
global_swap_tracking_flag = False
global_delivered_EPR_flag = False
printing_attempt_flag = False
printing_attempt_success_flag = False
printing_attempt_result_flag = False
global_go_to_sleep_flag  =False
global_delivery_printing_flag  = False
global_number_of_sleeping_sec_attempt = 0.1
global_number_of_sleeping_sec_attempt_full_fail = 0.1
global_number_of_sleeping_sec_attempt_full_suc = 10
global_number_of_sleeping_sec_attempt_half_suc = 10
global_number_of_sleeping_sec_expiration = 10
global_number_of_sleeping_sec_swap = 10
global_number_of_sleeping_sec_delivered = 10


mu = 1
mu_i_value = 1
experimenting_classical_communication = True
entanglement_generation_delay = True
having_cut_offs = True
considering_end_nodes_idle_time = False
each_R_path_repeaters = {1:{0:[0,1,2]},
                         2:{0:[0,1,2,3]},
                         3:{0:[0,1,2,3,4]},
                         4:{0:[0,1,2,3,4,5]},
                         5:{0:[0,1,2,3,4,5,6]},
                         6:{0:[0,1,2,3,4,5,6,7]},
                         7:{0:[0,1,2,3,4,5,6,7,8]},
                         8:{0:[0,1,2,3,4,5,6,7,8,9]},
                         9:{0:[0,1,2,3,4,5,6,7,8,9,10]}
                        }

each_R_path_links = {    1:{0:[(0,1),(1,2)]},
                         2:{0:[(0,1),(1,2),(2,3)]},
                         3:{0:[(0,1),(1,2),(2,3),(3,4)]},
                         4:{0:[(0,1),(1,2),(2,3),(3,4),(4,5)]},
                         5:{0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]},
                         6:{0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]},
                         7:{0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)]},
                         8:{0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9)]},
                         9:{0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10)]},
                        10:{0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11)]}
                        }
path_id_path_repeaters = {0:[0,1,2,3,4,5,6,7,8,9,10,11]}
path_id_path_links = {0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9)]}
# experiment_name = "random_placement"
# experiment_name="repeater_position"
experiment_name="equal"
L0_list = np.linspace(300,400,10)
L0_list = [200]
time_granularity_value = 5
passed_indx  =int(sys.argv[1])
# results_file_path = "results/random_placement_exp_time_step_5.csv"
# results_file_path = "results/one_repeater_plaement_testing_time_step_5.csv"
# results_file_path = "results/equal_distance_repeater_placement_as_distance.csv"
# results_file_path = "results/equal_distance_repeater_placement_cutoff_coherence_exp_time_step_5v2.csv"
results_file_path = "results/equal_distance_repeater_placement_coherence_cutoff_distance_color_map_exp_time_step_10.csv"
results_file_path = "results/equal_distance_repeater_placement_coherence_cutoff_testing_time_step_oneLe2e50km.csv"
results_file_path = "results/equal_distance_repeater_placement_coherence_cutoff_testing_time_step_oneLe2e100kmv5.csv"
results_file_path = "results/equal_distance_repeater_placement_cutoff_Le2e200km_time_step_5_testingv7.csv"
for iteration in range(1000000):
    for number_of_repeaters in [3]:
    # for number_of_repeaters in [6]:
        path_id_path_repeaters = each_R_path_repeaters[number_of_repeaters]
        path_id_path_links =each_R_path_links[number_of_repeaters] 
        each_path_source_destination = {0:[0,number_of_repeaters+1]}
        initial_fidelity  = 1
        
        
        for memory_max in [1]:
            
            for i, L0 in enumerate(L0_list):
                each_link_length,rep_loc = compute_edges(experiment_name,L0,number_of_repeaters)
                print("each_link_length,rep_loc",each_link_length,rep_loc)
                import pdb
                # pdb.set_trace()
                for j, pos in enumerate(rep_loc):
                    if pos>=0.1 or 1==1:
                        # pos = 0.5
                        if experiment_name =="repeater_position":
                            Le2e = L0_list[0]
                            L1 = pos*Le2e
                            L2 = Le2e - L1
                            L0 = pos
                            left_link_length = L1
                            each_link_length = {(0,1):L1,(1,2):L2,(2,3):left_link_length,
                                                    (3,4):left_link_length,
                                                   (4,5):left_link_length,
                                                   (5,6):left_link_length,
                                                    (6,7):left_link_length,
                                                   (7,8):left_link_length,
                                                   (8,9):left_link_length,
                                                   (9,10):left_link_length,
                                                   (10,11):left_link_length,
                                                   (11,12):left_link_length}
    
    
                        
                        each_path_memory_min = {0:0}
                        each_path_memory_max= {0:memory_max}
                        running_time = 50000000
                        # print(pos,each_link_length[(0,1)],each_link_length[(1,2)])
                       
                        scheme = "parallel"
                        for path_id in path_id_path_links:
                            if not having_cut_offs:
                                for coherecne_time in range(10,100,10):
                                    system = System(scheme,coherecne_time/1000,number_of_repeaters,path_id,running_time,1,1000,each_link_length,
                                                    each_path_memory_min,each_path_memory_max)
                                    system.main()
                                    e2e_rate = system.e2e_EPRs/running_time*1000000
                                    try:
                                        rate_of_generating_first_segment = mean(system.successfull_first_segment_geenration_times)
                                        number_of_successful = len(system.successfull_first_segment_geenration_times)
                                    except:
                                        rate_of_generating_first_segment= 0
                                        number_of_successful = 0
                                    if e2e_rate>0:
                                        S,r,mu_e2e,avg_e2e_f,e_x,e_z,e2e_f_including_end_nodes= system.sekret_key(path_id,number_of_repeaters,e2e_rate)
                                    else:
                                        S,r,mu_e2e,avg_e2e_f,e_x,e_z,e2e_f_including_end_nodes = 0,0,0,0,0,0,0
                                        
                                    if len(system.all_delivery_durations)!=0:
                                        avg_T = sum(system.all_delivery_durations)/len(system.all_delivery_durations)
                                    else:
                                        avg_T = 0
                                    line_items = [scheme,experimenting_classical_communication,
                                                                    having_cut_offs,
                                                                    number_of_repeaters,
                                                                    0,memory_max,running_time,
                                                                    e2e_rate,system.longest_link_duration,
                                                                   entanglement_generation_delay,path_id,
                                                                   S,r,avg_e2e_f,L0,avg_T,e_x,e_z,
                                                     system.expired_qubits_counter,
                                                 rate_of_generating_first_segment,
                                                     number_of_successful,experiment_name,iteration,i,j,
                                                  having_cut_offs,e2e_f_including_end_nodes,
                                                 coherecne_time,mu_e2e]
        
        
                                    
                                    
                                    # for repeater in  system.path_id_path_repeaters[system.path_id]:
                                    #     if e2e_rate>0:
                                    #         waiting_times = system.each_repeater_left_right_memories_waiting_times[system.path_id,repeater]
                                    #         left_q_idle_times = []
                                    #         right_q_idle_times = []
                                    #         for time in waiting_times:
                                    #             with open("results/parallel_each_instance_each_repeater_left_right_timev12.csv", 'a') as newFile:                                
                                    #                 newFileWriter = csv.writer(newFile)
                                    #                 newFileWriter.writerow([number_of_repeaters,L0,repeater,time[0],time[1]])
                
                                    #             left_q_idle_times.append(time[0])
                                    #             right_q_idle_times.append(time[1])
                                    #         if right_q_idle_times:
                                    #             avg_left_idle_time = mean(left_q_idle_times)
                                    #             avg_right_idle_time = mean(right_q_idle_times)
                                    #     else:
                                    #         avg_left_idle_time = -1
                                    #         avg_right_idle_time = -1
                                    #     line_items.append(str(avg_left_idle_time)+":"+str(avg_right_idle_time))
                                    print("scheme %s L %s cut_off %s R %s time %s delivered %s e2e EPRs and e2e_rate %s SKR %s avg_e2e_f %s avg_T %s"%(scheme,L0,
                                                                                                                                                "no_cut_off",number_of_repeaters,
                                                                                                                                                running_time,system.e2e_EPRs,
                                                                                                                                                e2e_rate,S,avg_e2e_f,
                                                                                                                                          avg_T))
                                    with open(results_file_path, 'a') as newFile:                                
                                        newFileWriter = csv.writer(newFile)
                                        newFileWriter.writerow([item for item in line_items])
                                        
                                    # if e2e_rate>0:
                                    #     for each_e2e_f in system.each_path_all_delivered_pairs_fidelity[system.path_id]:
                                    #         with open("results/parallel_each_instance_e2e_fv12.csv", 'a') as newFile:                                
                                    #             newFileWriter = csv.writer(newFile)
                                    #             newFileWriter.writerow([number_of_repeaters,L0,each_e2e_f])
                                    
                            else:
            #                         for cut_off in [4,6,8,12,16,100,40,50,200,0.5,0.05,60000000]:#in milliseconds
            #                         for cut_off in [0.01,0.02,0.04,0.1,0.2,0.4,0.8,1,2,4,8,10,20,40,80,100,200,400]:#in milliseconds
                                # for cut_off in [1,2,3,4,5,10,20,40,80,100,200,500,1000,2000,4000]:#in milliseconds
                                # for cut_off in range(1,100,5):#in milliseconds
                                # for coherecne_time in range(0.1,100,10):
                                τ_coh_list = np.logspace(-4.3,-2,30)
                                τ_coh_list =[0.01]
                                coherence_indx = -1
                                for coherecne_time in τ_coh_list:# in seconds
                                    coherence_indx+=1
                                    if coherence_indx==passed_indx:
                                    # for coherecne_time in [0.1]:# in seconds
                                        coherecne_time = coherecne_time*1000
                                        # for cut_off in range(10,500,5):
                                        # for cut_off in [0.05]:
                                        c = 2e5 # speed of light in fiber [km/s]
                                        # τ_cut_list = np.logspace(-0.5,2,20)*each_link_length[(0,1)]/c/2 # cutoff [sec]
                                        Le2e = 200
                                        τ_cut_list = np.logspace(0,4,41)*Le2e/(2*c) # cutoff [sec]
                                        # for cut_off in τ_cut_list:#in seconds
                                        for cut_off in [0.0019905358527674867,0.02,0.03,0.04,0.05,0.06294627058970838, 0.07924465962305571, 0.09976311574844403]:
                                        # for cut_off in τ_cut_list:
                                            
                                            cut_off = cut_off*1000
                                            cut_off = cut_off*1000
                                            starting_time = time.time()
                                            system = System(scheme,coherecne_time/1000,number_of_repeaters,path_id,running_time,1,cut_off,each_link_length,
                                                            each_path_memory_min,each_path_memory_max)
                                            system.main()
                                            e2e_rate = system.e2e_EPRs/running_time*1000000
                                            if e2e_rate>0:
                                                S,r,mu_e2e,avg_e2e_f,e_x,e_z,e2e_f_including_end_nodes= system.sekret_key(path_id,number_of_repeaters,e2e_rate)
                                            else:
                                                S,r,mu_e2e,avg_e2e_f,e_x,e_z,e2e_f_including_end_nodes = 0,0,0,0,0,0,0     
                                            try:
                                                rate_of_generating_first_segment = mean(system.successfull_first_segment_geenration_times)
                                                number_of_successful = len(system.successfull_first_segment_geenration_times)
                                            except:
                                                rate_of_generating_first_segment = 0
                                                number_of_successful = 0
                                            if len(system.all_delivery_durations)!=0:
                                                avg_T = sum(system.all_delivery_durations)/len(system.all_delivery_durations)
                                            else:
                                                avg_T = 0
                                            line_items = [scheme,experimenting_classical_communication,
                                                                        having_cut_offs,
                                                                        number_of_repeaters,
                                                                        cut_off,memory_max,running_time,
                                                                        e2e_rate,system.longest_link_duration,
                                                                       entanglement_generation_delay,path_id,
                                                                       S,r,avg_e2e_f,L0,avg_T,e_x,e_z,
                                                         system.expired_qubits_counter,
                                                     rate_of_generating_first_segment,
                                                         number_of_successful,experiment_name,iteration,i,j,having_cut_offs,
                                                         e2e_f_including_end_nodes,coherecne_time,mu_e2e]
                    
                                            # for repeater in  system.path_id_path_repeaters[system.path_id]:
                                                # if e2e_rate>0:
                                                #     waiting_times = system.each_repeater_left_right_memories_waiting_times[system.path_id,repeater]
                                                #     left_q_idle_times = []
                                                #     right_q_idle_times = []
                                                #     for w_time in waiting_times:
                                                #         # with open("results/parallel_each_instance_each_repeater_left_right_time_withcutoff_values.csv", 'a') as newFile:                                
                                                #         #     newFileWriter = csv.writer(newFile)
                                                #         #     newFileWriter.writerow([number_of_repeaters,L0,repeater,
                                                #         #                             w_time[0],w_time[1],cut_off])
                    
                                                #         left_q_idle_times.append(w_time[0])
                                                #         right_q_idle_times.append(w_time[1])
                                                #     if right_q_idle_times:
                                                #         avg_left_idle_time = mean(left_q_idle_times)
                                                #         avg_right_idle_time = mean(right_q_idle_times)
                                                #         avg_left_idle_time = avg_left_idle_time/1000000
                                                #         avg_right_idle_time = avg_right_idle_time/1000000
                                                #         print("repeater %s avg_left_idle_time %s avg_right_idle_time %s max left  %s min left %s "%(repeater,avg_left_idle_time,
                                                #                                                                                         avg_right_idle_time,max(left_q_idle_times),
                                                #                                                                                        min(left_q_idle_times)))
                                                        
                                                # else:
                                                #     avg_left_idle_time = -1
                                                #     avg_right_idle_time = -1
                                                # line_items.append(str(avg_left_idle_time)+":"+str(avg_right_idle_time))
                                            print("scheme %s L %s cut_off %s R %s time %s delivered %s e2e EPRs and e2e_rate %s SKR %s avg_e2e_f %s avg_T %s ms expired #s %s"%(scheme,L0,
                                                                                                                                                        cut_off,number_of_repeaters,
                                                                                                                                                        running_time,system.e2e_EPRs,
                                                                                                                                                        e2e_rate,S,avg_e2e_f,
                                                                                                                                                  avg_T/1000,
                                                                                                                                                  system.expired_qubits_counter))
                                            current_time = time.time()
                                            processing_time_in_seconds = int(current_time -starting_time)
                                            print("it took %s seconds for this round"%(processing_time_in_seconds))
                                            time.sleep(1)
                                            if system.one_successful_round_times:
                                                avg_success_T = sum(system.one_successful_round_times)/len(system.one_successful_round_times)
                                            else:
                                                avg_success_T = -1
                                            Ns = len(system.one_successful_round_times)
                                            Nf=len(system.one_failed_round_times)
                                            if system.one_failed_round_times:
                                                avg_fail_T = sum(system.one_failed_round_times)/len(system.one_failed_round_times)
                                            else:
                                                avg_fail_T = -1
                                            line_items.append(Ns)
                                            line_items.append(avg_success_T)
                                            line_items.append(Nf)
                                            line_items.append(avg_fail_T)
                                        
                                            import pdb
                                            #pdb.set_trace()
                                            # if global_go_to_sleep_flag:
                                            #     time.sleep(30)
                                            with open(results_file_path, 'a') as newFile:                                
                                                newFileWriter = csv.writer(newFile)
                                                newFileWriter.writerow([item for item in line_items])
                    
                                            # if e2e_rate>0:
                                            #     for each_e2e_f in system.each_path_all_delivered_pairs_fidelity[system.path_id]:
                                            #         with open("results/parallel_each_instance_e2e_with_cutoff_values.csv", 'a') as newFile:                                
                                            #             newFileWriter = csv.writer(newFile)
                                            #             newFileWriter.writerow([number_of_repeaters,L0,each_e2e_f,cut_off])
                
                                        


# In[4]:





# In[ ]:





# In[ ]:


# global_printing_flag = False

# each_R_path_repeaters = {1:{0:[0,1,2]},
#                          2:{0:[0,1,2,3]},
#                          3:{0:[0,1,2,3,4]},
#                          4:{0:[0,1,2,3,4,5]},
#                          5:{0:[0,1,2,3,4,5,6]},
#                          6:{0:[0,1,2,3,4,5,6,7]}
#                         }

# each_R_path_links = {    1:{0:[(0,1),(1,2)]},
#                          2:{0:[(0,1),(1,2),(2,3)]},
#                          3:{0:[(0,1),(1,2),(2,3),(3,4)]},
#                          4:{0:[(0,1),(1,2),(2,3),(3,4),(4,5)]},
#                          5:{0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]},
#                          6:{0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]}
#                         }
# path_id_path_repeaters = {0:[0,1,2,3,4,5,6,7,8]}
# path_id_path_links = {0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)]}
# for run in range(100):
#     for L0 in [100,50,20,200]:
#         for number_of_repeaters in [1]:
#             path_id_path_repeaters = each_R_path_repeaters[number_of_repeaters]
#             path_id_path_links =each_R_path_links[number_of_repeaters] 
#             each_path_source_destination = {0:[0,number_of_repeaters+1]}
#             initial_fidelity  = 0.95
#             experimenting_classical_communication = True
#             entanglement_generation_delay = True
#             having_cut_offs = True
#             results_file_path = "results/asynch_repeater_chain_results_repeater_position_expv3.csv"
#             # for memory_max in [1,2,4,6,8,10,14,16]:
#             #     for cut_off in [50,60,70,80,90,100,110,120,130,140,150,200,300]:
#             #         for left_link_length in [4,6,8,10,12,14,16,18,20]:
#             for memory_max in [1]:
#         #         for left_link_length in [1,50,100,5,10,20,30,40,50,15,25,35,45,55,60,65,70,75,80,90,200,300,400,500,600,700,800]:
#         #         for left_link_length in [10]:



#     #             L0 = 100 # e2e distance [km]
#                 c = 2e5 # speed of light [km/s]
#                 τ_coh = 0.1

#                 δ = 0.2
#                 rep_loc = np.linspace(δ,1-δ,30)
#                 inv_rate_par = np.zeros(len(rep_loc))
#                 f_memory_par = np.zeros(len(rep_loc))
#                 # inv_rate_mc = np.zeros(len(rep_loc))
#                 # inv_rate_old = np.zeros(len(rep_loc))
#                 # memory_time_old = np.zeros(len(rep_loc))
#                 inv_rate_seq = np.zeros(len(rep_loc))
#                 f_memory_seq = np.zeros(len(rep_loc))
#                 for i, pos in enumerate(rep_loc):
#                     L1 = pos*L0
#                     L2 = L0 - L1
#                     left_link_length = L1
#                     each_link_length = {(0,1):L1,(1,2):L2,(2,3):left_link_length,
#                                         (3,4):left_link_length,
#                                        (4,5):left_link_length,
#                                        (5,6):left_link_length,
#                                        (6,7):left_link_length,
#                                        (7,8):left_link_length}
#                     each_path_memory_min = {0:0}
#                     each_path_memory_max= {0:memory_max}
#                     running_time = 100000000
#                     scheme = "parallel"
#                     if not having_cut_offs:
#                         system = System(scheme,running_time,1,1000,each_link_length,
#                                         each_path_memory_min,each_path_memory_max)
#                         system.main()
#                         e2e_rate = system.e2e_EPRs/running_time
#                         print("*****!!!!**** scheme %s L %s cut_off %s M %s time %s delivered %s e2e EPRs and e2e_rate %s"%(scheme,left_link_length,"no_cut_off",memory_max,running_time,system.e2e_EPRs,e2e_rate))
#                         with open(results_file_path, 'a') as newFile:                                
#                             newFileWriter = csv.writer(newFile)
#                             newFileWriter.writerow([scheme,experimenting_classical_communication,
#                                                     having_cut_offs,
#                                                     number_of_repeaters,
#                                                     left_link_length,False,memory_max,running_time,
#                                                     e2e_rate,system.longest_link_duration,
#                                                    entanglement_generation_delay,pos,L1,L2,L0,run])
#                     else:
#                         for cut_off in [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000, 30000, 40000, 60000]:
#         #                 for cut_off in [500]:
#                             system = System(scheme,running_time,1,cut_off,each_link_length,
#                                             each_path_memory_min,each_path_memory_max)
#                             system.main()
#                             e2e_rate = system.e2e_EPRs/running_time
#                             print("*****!!!!**** scheme %s L %s cut_off %s M %s time %s delivered %s e2e EPRs and e2e_rate %s"%(scheme,left_link_length,cut_off,memory_max,running_time,system.e2e_EPRs,e2e_rate))
#                             with open(results_file_path, 'a') as newFile:                                
#                                 newFileWriter = csv.writer(newFile)
#                                 newFileWriter.writerow([scheme,experimenting_classical_communication,
#                                                         having_cut_offs,
#                                                         number_of_repeaters,
#                                                         left_link_length,cut_off,memory_max,running_time,
#                                                         e2e_rate,system.longest_link_duration,
#                                                        entanglement_generation_delay,pos,L1,L2,L0,run])
            


# In[29]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




