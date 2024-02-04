#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import time
import csv
import networkx as nx
import numpy as np
import math


# In[3]:


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
    def __init__(self,experimenting_classical_communication_flag,repeater,destination,path_id,message_type,left_node,right_node,memory_id_left,memory_id_right,left_age,right_age,clock_counter,link_duration,success_flag):
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
    def __init__(self,mode,path_id,end_time,q_value,cut_off,each_link_length,each_path_memory_min,each_path_memory_max):
        self.mode = mode
        self.considering_end_nodes_idle_time = considering_end_nodes_idle_time
        self.path_id = path_id
        self.end_time = end_time
        self.experimenting_classical_communication = experimenting_classical_communication
        self.having_cut_offs = having_cut_offs
        self.entanglement_generation_delay = entanglement_generation_delay
        self.q_value = q_value
        self.e2e_EPRs = 0
        self.clock_counter = 0
        self.expired_qubits_counter = 0
        self.synchronous_flag = False
        self.initial_fidelity = initial_fidelity 
        self.cut_off= cut_off
        self.all_delivery_durations = []
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
        self.each_path_swap_on_source_node_qubit_arrived={}
        self.each_path_swap_on_source_node_qubit_id={}
        self.each_path_swap_on_end_node_qubit_id = {}
        self.longest_link_duration = 0
        self.message_channel = []
        self.swap_list = {}
        self.swap_start_sending_time_list = {}
        self.failed_at_least_one_link = False
        self.each_path_all_delivered_pairs_fidelity = {}
        self.each_path_all_delivered_pairs_fidelity_including_end_nodes = {}
        self.each_path_each_repeater_swaped_memory_ages ={}
        self.t_coh = 0.1
        self.F = {}
        self.each_repeater_left_right_memories_waiting_times = {}
        self.each_repeater_left_right_memory_age_tracking_flag = {}
        self.mu_i = {}
        self.mu = 0.97
        self.last_generated_expiration_message = 0
        self.one_successful_round_times=[]
        self.last_delivered_e2e_epr_time = 0
        self.last_failed_time = 0
        self.last_be_geenrated_time = 0
        self.one_failed_round_times = []
        self.expiration_message_arriving_time = 0
        self.time_granularity_value = time_granularity_value
        self.global_expiration_flag = False
#         for repeater in [1]:
#             self.F[path_id,repeater]= 1
#             self.mu_i[path_id].append(0.97)
            
        
#         for path_id in self.path_id_path_links:
        self.each_path_swap_on_source_node_qubit_arrived[self.path_id]=False
        self.each_path_swap_on_source_node_qubit_id[self.path_id] = -1
        self.each_path_swap_on_end_node_qubit_id[self.path_id] = -1
        for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
            self.F[self.path_id,repeater]= 1
            try:
                self.mu_i[self.path_id].append(0.97)
            except:
                self.mu_i[self.path_id] = [0.97]
            self.swap_list[repeater]=[]
            self.swap_start_sending_time_list[repeater] = []   

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
#         for path_id,path_links in self.path_id_path_links.items():
        repeater_index = 0
        for link in self.path_id_path_links[self.path_id]:
            length = self.each_link_length[link]
            link_lengths.append(length)
            repeater_index+=1
        longest_link = max(link_lengths)
        self.longest_link_duration = int((1.44*longest_link)/ 299792*1000000)
            
#         for path_id,path_links in self.path_id_path_links.items():
        for link in self.path_id_path_links[self.path_id]:
            for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
                length = self.each_link_length[link]
                link_success_p = 10**(-0.2*length/10)
#                     link_success_p = 1
                link_duration = int((1.44*length)/ 299792*1000000)
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
        # print("**************************** length from %s to %s is length %s= %s "%(source,repeater,shortest_path_length1,shortest_path_length1/2e5*1000000))
        return shortest_path_length1,source
    
        
                        
                        
    
                    


    def copy_left_instanse_to_right_instance(self,left_memory_object,right_memory_object):
        right_memory_object.attempt_flag = left_memory_object.attempt_flag
        right_memory_object.synchronous_flag = left_memory_object.synchronous_flag
        right_memory_object.attempt_start_time = left_memory_object.attempt_start_time
        right_memory_object.attempt_finishing_time = left_memory_object.attempt_finishing_time
        right_memory_object.initial_fidelity = left_memory_object.initial_fidelity
        right_memory_object.link_duration = left_memory_object.link_duration
        right_memory_object.each_time_slot_duration = left_memory_object.each_time_slot_duration
        
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
        
        if left_memory_object.attempt_flag:#we have attempted before lets check if it is done

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
            if (EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and 
                link[0] not in self.each_path_source_destination[self.path_id]) or (
                right_EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and 
                link[1] not in self.each_path_source_destination[self.path_id]):
                    expired_flag = True
                    self.expired_qubits_counter+=1
                    self.last_generated_expiration_message = current_clock_counter
                    if (EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and  
                        link[0] not in self.each_path_source_destination[self.path_id]):
                        left_memory_object.expired = True
                        distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[0])
                    elif (right_EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and 
                          link[1] not in self.each_path_source_destination[self.path_id]):
                        right_memory_object.expired = True
                        distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[1])
                    time_to_source = int(distance_from_source/2e5*1000000)
                    if not self.global_expiration_flag:
                        self.global_expiration_flag = True
                        if printing_qubit_expiration_flag:
                            print("we set the arriving time duration for expiration message to ",time_to_source)
                        self.expiration_message_arriving_time = current_clock_counter+time_to_source
                    if printing_qubit_expiration_flag:
                        print("0!!!!!!!!!!!!!!!!!!!!unfortunately the qubit got aged at link %s %s  age %s cut_off %s at time %s will take %s %s "%(left_memory_object.repeater_id,
                                                                                                                                                    left_memory_object.entangled_node,left_memory_object.EPR_age, 
                                                                                                                                                    self.cut_off,current_clock_counter,self.expiration_message_arriving_time,
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
            if (EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and 
                link[0] not in self.each_path_source_destination[self.path_id]) or (
                right_EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and 
                link[1] not in self.each_path_source_destination[self.path_id]):
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
                if (EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and  
                        link[0] not in self.each_path_source_destination[self.path_id]):
                    left_memory_object.expired = True
                    distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[0])
                elif (right_EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and 
                      link[1] not in self.each_path_source_destination[self.path_id]):
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

#             else:
#                 print("left_memory_object.attempt_flag off for link %s no qubit has been expired with ages %s %s"%(link,left_memory_object.EPR_age,right_memory_object.EPR_age))

        # self.copy_left_instanse_to_right_instance(left_memory_object,right_memory_object)
        
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
        
        
        
                                                                                                                
                  
        
            
    def attempt_to_generate(self,link,memory_id,current_clock_counter):
#         print("going to attempt!")
        left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id]
        right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[1],"left",memory_id]
        left_node_memory_exist_flag = self.each_repeater_left_right_memory_exist_flag[self.path_id,link[0],"right",memory_id]
        right_node_memory_exist_flag = self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id]
        generated_flag = False
        if printing_qubit_aging_flag:
            print(" 1 time clock %s   link (%s,%s) last update (%s, %s) age track flag(%s,%s) got aged by one new age %s, %s FLAG %s "%(current_clock_counter,link[0],link[1],
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
            if (EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and 
                link[0] not in self.each_path_source_destination[self.path_id]) or (
                right_EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and 
                link[1] not in self.each_path_source_destination[self.path_id]):
                    expired_flag = True
                    self.expired_qubits_counter+=1
                    self.last_generated_expiration_message = current_clock_counter
                    if (EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and  
                        link[0] not in self.each_path_source_destination[self.path_id]):
                        left_memory_object.expired = True
                        distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[0])
                    elif (right_EPR_age >=self.each_link_cut_off[link] and self.having_cut_offs and 
                          link[1] not in self.each_path_source_destination[self.path_id]):
                        right_memory_object.expired = True
                        distance_from_source,_ = self.shortest_path_to_end_nodes_length(link[1])
                    time_to_source = int(distance_from_source/2e5*1000000)
                    if not self.global_expiration_flag:
                        self.global_expiration_flag = True
                        if printing_qubit_expiration_flag:
                            print("we set the arriving time duration for expiration message to ",time_to_source)
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
                    generated_flag = True
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
                
        return generated_flag
            
    def remove_all_generated_link_level_eprs(self,memory_id):
        global global_printing_flag
        if global_printing_flag:
            print("we are removing whatever link that has been generated so far")
        
        
        for link in self.path_id_path_links[path_id]:
            left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id]
            right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[1],"left",memory_id]
            left_memory_object.attempt_flag = False
            left_memory_object.EPR_age = -1
            self.each_repeater_left_right_memory_exist_flag[self.path_id,link[0],"right",memory_id] = False
            self.each_repeater_left_right_memory_exist_flag[self.path_id,link[1],"left",memory_id]  = False
            self.copy_left_instanse_to_right_instance(left_memory_object,right_memory_object)
    def send_a_message(self,repeater,path_id,message_type,left_node_qubit_age,
                       right_node_qubit_age,left_node,right_node,
                       memory_id_left,memory_id_right,
                       current_clock_counter,success_flag):
        length,destination = self.shortest_path_to_end_nodes_length(repeater)
        link_duration = int((1.44*length)/ 299792*1000000)
        
        message = Message(self.experimenting_classical_communication,repeater,destination,path_id,
                          message_type,left_node,right_node,memory_id_left,memory_id_right,left_node_qubit_age,
                          right_node_qubit_age,current_clock_counter,link_duration,success_flag)
        if global_printing_flag:
            print("we added the message sent from %s arrive at %s to the channel all %s"%(message.sender,message.arriving_time,len(self.message_channel)))
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
        if global_printing_flag:
            print("swap result arrived from %s at %s on qubits with ages %s %s !rec./unrec. swap results: %s "%(message.sender,clock_counter,message.left_qubit_age,message.right_qubit_age,self.swap_list))
        if message.left_node==self.each_path_source_destination[message.path_id][0]:
#             print("************************************ this arrived swap is entangeled with the source node")
            self.each_path_swap_on_source_node_qubit_arrived[message.path_id] = True
            self.each_path_swap_on_source_node_qubit_id[message.path_id] = message.memory_id_right
        if message.right_node==self.each_path_source_destination[message.path_id][1]:
            self.each_path_swap_on_end_node_qubit_id[message.path_id]=message.memory_id_right
    def swap_operation(self,current_clock_counter,repeater):
        global global_printing_flag
        left_side_memory_ages = []
        right_side_memory_ages = []
        left_memory_objects = []
        right_memory_objects = []
        for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
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
        for old_left in left_side_memory_ages:
            swapped_with_one_EPR = False
            for left_memory_object in left_memory_objects:
                if left_memory_object.EPR_age ==old_left:
                    for old_right in right_side_memory_ages:
                        for right_memory_object in right_memory_objects:
                            if right_memory_object.EPR_age==old_right:
                                swapped_with_one_EPR =True
                                random_value  = random.uniform(0,1)
#                                 print("turning the memory flag on right of %s to False as we did swap at %s "%(repeater,repeater))
#                                 print("turning the memory flag on left of %s to False as we did swap at %s "%(repeater,repeater))
                                self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"right",right_memory_object.memory_id] =False
                            
                                self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"left",left_memory_object.memory_id] =False
                                self.failed_at_least_one_link = False
                                if random_value <=self.q_value:
                                    
                                    left_entangled_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,left_memory_object.entangled_node,"right",left_memory_object.entangled_memory_id]
                                    right_entangled_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,right_memory_object.entangled_node,"left",right_memory_object.entangled_memory_id]

                                    
                                    self.send_a_message(repeater,self.path_id,"swap_result",left_memory_object.EPR_age,
                                                        right_memory_object.EPR_age,
                                                        left_memory_object.entangled_node,
                                                        right_memory_object.entangled_node,
                                                        left_memory_object.memory_id,
                                                        right_memory_object.memory_id,
                                                        current_clock_counter,True)
                                    left_memory_object.attempt_flag = False
                                    right_memory_object.attempt_flag = False
                                    left_memory_object.EPR_age = -1
                                    right_memory_object.EPR_age = -1
#repeater
                                elif random_value > self.q_value:
                                    self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"right",right_memory_object.memory_id] =False
                                    self.each_repeater_left_right_memory_exist_flag[self.path_id,repeater,"left",left_memory_object.memory_id] =False
                                    self.each_repeater_left_right_memory_exist_flag[self.path_id,left_memory_object.entangled_node,"right",left_memory_object.memory_id] =False
                                    self.each_repeater_left_right_memory_exist_flag[self.path_id,right_memory_object.entangled_node,"left",right_memory_object.memory_id] =False
                                    left_entangled_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,left_memory_object.entangled_node,"right",left_memory_object.entangled_memory_id]
                                    right_entangled_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,right_memory_object.entangled_node,"left",right_memory_object.entangled_memory_id]

                                    left_entangled_node_memory_object.attempt_flag = False
                                    right_entangled_node_memory_object.attempt_flag = False
                                    left_memory_object.attempt_flag = False
                                    right_memory_object.attempt_flag = False                    
                                    left_memory_object.EPR_age = -1
                                    right_memory_object.EPR_age = -1
    def check_swap_results_arrived(self,clock_counter):
#         if global_printing_flag:
#             print("we are checking if there is any arrived message",len(self.message_channel))
#             for message in self.message_channel:
#                 print("there is a message from ",message.sender,message.arriving_time,clock_counter,len(self.message_channel))
        messages_to_be_removed = []
        for message in self.message_channel:
#             print("for message from ",message.sender,message.arriving_time,clock_counter,len(self.message_channel))
            if message.arriving_time<=clock_counter and not message.processing_flag:
                message.processing_flag = True
                if message.type =="cancelling":
                    self.receive_a_fizzling_message(message,clock_counter)
                elif message.type =="swap_result":
                    self.receive_a_swap_result_message(message,clock_counter)
#                 print("before removing message from channel ",len(self.message_channel))
                
                    messages_to_be_removed.append(message)
            
#                 print("after removing message from channel ",len(self.message_channel))
        for message in messages_to_be_removed:
        
            left_age=message.left_qubit_age
            right_age = message.right_qubit_age
            repeater = message.sender
            try:
                self.each_path_each_repeater_swaped_memory_ages[self.path_id,repeater].append((left_age,right_age))
            except:
                self.each_path_each_repeater_swaped_memory_ages[self.path_id,repeater] = [(left_age,right_age)]
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
    
        
        
    def check_all_swaps(self,clock_counter):
        missing_one_swap = False
        for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
            if len(self.swap_list[repeater])>0:
                if self.swap_list[repeater][0]==0:
                    missing_one_swap= True
            else:
                missing_one_swap= True
                
        if not missing_one_swap:
            if global_printing_flag:
                print("********************* we delivered one e2e EPR pair at time %s *********messages in channel %s***************"%(clock_counter,len(self.message_channel)))
                time.sleep(5)

            self.one_successful_round_times.append(clock_counter - self.last_delivered_e2e_epr_time)
            self.last_delivered_e2e_epr_time = clock_counter
            self.last_failed_time = clock_counter
            self.global_expiration_flag = False
            self.e2e_EPRs+=1
            each_repeater_left_right_memory_idle_time = {}
            for path_repeater,times in self.each_path_each_repeater_swaped_memory_ages.items():
                key_path_id =path_repeater[0]
                repeater = path_repeater[1]
                if self.path_id==key_path_id:
                    time = times[0]
                    each_repeater_left_right_memory_idle_time[repeater] = time
            for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
                if len(self.each_path_each_repeater_swaped_memory_ages[self.path_id,repeater])==1:
                    self.each_path_each_repeater_swaped_memory_ages[self.path_id,repeater] = []
                else:
                    self.each_path_each_repeater_swaped_memory_ages[self.path_id,repeater] = self.each_path_each_repeater_swaped_memory_ages[self.path_id,repeater][1:]
            f_e2e,e2e_fidelity=self.compute_f_e2e(each_repeater_left_right_memory_idle_time)
            try:
                self.each_path_all_delivered_pairs_fidelity[self.path_id].append(f_e2e)
            except:
                self.each_path_all_delivered_pairs_fidelity[self.path_id] = [f_e2e]

            try:
                self.each_path_all_delivered_pairs_fidelity_including_end_nodes[self.path_id].append(e2e_fidelity)
            except:
                self.each_path_all_delivered_pairs_fidelity_including_end_nodes[self.path_id] = [e2e_fidelity]
                
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
            
    
    def h(self,e_x):
        return -e_x*math.log(e_x)-(1-e_x)*math.log(1-e_x)
    def compute_f_e2e(self,each_repeater_left_right_memory_idle_time):
        if global_printing_flag:
            print("each_repeater_left_right_memory_idle_time ",each_repeater_left_right_memory_idle_time)
        product = 1
        sum_of_time = 0
        for repeater,left_right_time in each_repeater_left_right_memory_idle_time.items():
            if global_printing_flag:
                tau = 2*self.longest_link_duration
                if printing_qubit_aging_flag:
                    print("swap on repeater %s happened on qubits with age %s and %s 2_tau is %s have %s Rs "%(repeater,
                                                                                                left_right_time[0],
                                                                                                left_right_time[1],
                                                                                                tau,
                                                                                                len(list(each_repeater_left_right_memory_idle_time.keys()))))
            sum_of_time = sum_of_time+left_right_time[0]+left_right_time[1]#(4,10)
            try:
                self.each_repeater_left_right_memories_waiting_times[self.path_id,repeater].append(left_right_time)
            except:
                self.each_repeater_left_right_memories_waiting_times[self.path_id,repeater] = [left_right_time]
            product = product*(2*self.F[self.path_id,repeater]-1)
        source = self.each_path_source_destination[self.path_id][0]
        destination = self.each_path_source_destination[self.path_id][1]
        source_right_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,source,"right",0]
        destination_left_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,destination,"left",0]

        source_left_right_time = (0,source_right_memory_object.EPR_age)
        destination_left_right_time = (destination_left_memory_object.EPR_age,0)
        if global_printing_flag:
            T= 0
            e2e = 0
            for link in self.path_id_path_links[self.path_id]:
                length = self.each_link_length[link]
                link_duration = int(length/2e5*1000000)
                e2e = e2e+link_duration
                T = T+2*link_duration
            analytic_bob = e2e
            analytic_alice = T
            print("end node source destination idle time %s %s analytic idle time %s %s "%(source_right_memory_object.EPR_age,
                                                                                        analytic_alice,
                                                                                       destination_left_memory_object.EPR_age,
                                                                                        analytic_bob
                                                                                           
                                                                                      ))
        try:
            self.each_repeater_left_right_memories_waiting_times[self.path_id,source].append(source_left_right_time)
        except:
            self.each_repeater_left_right_memories_waiting_times[self.path_id,source] = [source_left_right_time]
        try:
            self.each_repeater_left_right_memories_waiting_times[self.path_id,destination].append(destination_left_right_time)
        except:
            self.each_repeater_left_right_memories_waiting_times[self.path_id,destination] = [destination_left_right_time]
        if self.considering_end_nodes_idle_time:
            sum_of_time = sum_of_time+source_left_right_time[1]+destination_left_right_time[0]


        sum_of_time_for_fidelity = sum_of_time+source_left_right_time[1]+destination_left_right_time[0]
        t_all = (sum_of_time_for_fidelity/self.t_coh)/1000000
        exp = np.exp(-t_all)
        e2e_fidelity=1/2+1/2*(exp)
        
        t = (sum_of_time/self.t_coh)/1000000
        exp = np.exp(-t)
        f_e2e=1/2+1/2*(exp)
        if global_printing_flag:
            print("product %s sum T %s /t_coh %s exp %s fidelity %s "%(product,sum_of_time,t,exp,f_e2e))
        return f_e2e,e2e_fidelity
    
    
    def sekret_key(self,path_id,n,rate):
        # avg_e2e_f = sum(self.each_path_all_delivered_pairs_fidelity[path_id])/len(self.each_path_all_delivered_pairs_fidelity[path_id])
        # mu_e2e=self.mu**n*np.prod(self.mu_i[path_id])
        # F_e2e = mu_e2e*avg_e2e_f+(1-mu_e2e)/4
        # e_x =(1+mu_e2e)/2-mu_e2e*avg_e2e_f
        # e_z=(1-mu_e2e)/2
        # r= 1-self.h(e_x)-self.h(e_z)
        # S = rate*r
        # return S,r,avg_e2e_f
        avg_e2e_f = sum(self.each_path_all_delivered_pairs_fidelity[path_id])/len(self.each_path_all_delivered_pairs_fidelity[path_id])
        avg_e2e_f_including_end_nodes = sum(self.each_path_all_delivered_pairs_fidelity_including_end_nodes[path_id])/len(self.each_path_all_delivered_pairs_fidelity_including_end_nodes[path_id])

        mu_e2e=(self.mu**n)*np.prod(self.mu_i[self.path_id])
        
        e_x =(1+mu_e2e)/2-(mu_e2e*avg_e2e_f)
        e_z=(1-mu_e2e)/2
        r= 1-self.h(e_x)-self.h(e_z)
        S = rate*r
        return S,r,avg_e2e_f,e_x,e_z,avg_e2e_f_including_end_nodes
    def none_expired(self):
        source= self.each_path_source_destination[self.path_id][0]
        end = self.each_path_source_destination[self.path_id][1]
        for link in self.path_id_path_links[self.path_id]:
            for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
                left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id]
                right_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[1],"left",memory_id]

                if link[0] == source:
                    if right_memory_object.expired:
                        return False
                elif link[1] == end:
                    if left_memory_object.expired:
                        return False
                else:
                    if left_memory_object.expired:
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
        self.global_expiration_flag = False
        if global_printing_flag:
            print("***************************we maid all memories free *****************")
            
        
    def main(self):
        self.e2e_EPRs = 0
        current_clock_counter = 0
        while(current_clock_counter <= self.end_time):
            
            if current_clock_counter%100000<=self.time_granularity_value:
                print("mode %s cut_off %s link %s M %s clock %s End %s "%(self.mode,self.cut_off,self.each_link_length[(0,1)],self.each_path_memory_max[0],current_clock_counter,self.end_time))
            
            path_links = self.path_id_path_links[self.path_id]
            expired_qubit = False
            for link in path_links:
                if global_printing_flag:
                    print("***************************************************************** %s,%s ****************************"%(link[0],link[1]))
                # time.sleep(3)
                if not expired_qubit:
                    if global_printing_flag:
                        print("time %s for link %s "%(current_clock_counter,link))
                    for memory_id in range(self.each_path_memory_min[path_id],self.each_path_memory_max[path_id]):
                        left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[0],"right",memory_id]
                        right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[1],"left",memory_id]
                        generated_a_link = False
    
                        while(current_clock_counter<= self.end_time and not generated_a_link and not expired_qubit):
                            generated_a_link = self.attempt_to_generate(link,memory_id,current_clock_counter)
                            # if global_printing_flag:
                            #     print("we will preform swaps ")
                            for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
                                self.swap_operation(current_clock_counter,repeater)
                            self.check_swap_results_arrived(current_clock_counter)
                            self.check_all_swaps(current_clock_counter)
                            for link2 in path_links:
                                self.update(link2,memory_id,current_clock_counter)
                            if not self.none_expired():
                                expired_qubit = True
                                if global_printing_flag:
                                    print("how long it takes to receive the expiration message ",self.expiration_message_arriving_time)
                                    print("we have one expired qubit. Lets move the time to the point the expiration message arrives at sender ",current_clock_counter,self.expiration_message_arriving_time,self.expiration_message_arriving_time)
                                current_clock_counter = self.expiration_message_arriving_time
                                # if max(self.last_generated_expiration_message+self.e2e_communication_duration,self.last_be_geenrated_time+self.e2e_communication_duration) <= clock_counter:
                                #if self.last_generated_expiration_message+self.e2e_communication_duration <= current_clock_counter:
                                # if self.expiration_message_arriving_time<=clock_counter:
                                    #print("lets free all memories and start a new round")
                                if global_go_to_sleep_flag:
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
                                        self.last_generated_expiration_message = current_clock_counter
                            if not expired_qubit:
                                current_clock_counter+=self.time_granularity_value
                            if current_clock_counter%100000<=self.time_granularity_value:
                                print("mode %s cut_off %s link %s M %s clock %s End %s "%(self.mode,self.cut_off,self.each_link_length[(0,1)],self.each_path_memory_max[0],current_clock_counter,self.end_time))
                        # if generated_a_link:
                        #     print("we generated the link %s at time %s "%(link,current_clock_counter))
                        #     time.sleep(5)

            if not expired_qubit:
                current_clock_counter+=self.time_granularity_value

                            
    #                         for link2 in path_links:
    #                             expired_qubit = self.update(link2,memory_id,current_clock_counter)
    #                             if expired_qubit:
    #                                 if global_qubit_expired_printing_flag:
    #                                     print("qubit expired at link %s we move time forward and make memories free"%(link2))
    #                                     time.sleep(2)
    #                                 current_clock_counter = current_clock_counter+self.expiration_message_arriving_time
    #                                 for link3 in self.path_id_path_links[self.path_id]:
    #                                     for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
    #                                         left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link3[0],"right",memory_id]
    #                                         left_memory_object.generated_flag = False
    #                                         left_memory_object.expired = False
    #                                         self.each_path_swap_on_source_node_qubit_arrived[self.path_id]=False
    #                                         self.each_path_swap_on_source_node_qubit_id[self.path_id] = -1
    #                                         self.each_path_swap_on_end_node_qubit_id[self.path_id] = -1
    #                                         self.make_all_memories_free()
    #                                         self.all_generated_flag = False
    #                                         self.swap_operation_is_performed_flag = False
    #                                         self.last_generated_expiration_message = current_clock_counter
    #                                         self.one_failed_round_times.append(current_clock_counter-self.last_failed_time)
    #                                         self.last_failed_time = current_clock_counter
    #                                         self.last_delivered_e2e_epr_time = current_clock_counter
                                    
                                    
    #                                 break
    #                         current_clock_counter+=self.time_granularity_value
    #                         if current_clock_counter%100000<=self.time_granularity_value:
    #                             print("mode %s cut_off %s link %s M %s clock %s End %s "%(self.mode,self.cut_off,self.each_link_length[(0,1)],self.each_path_memory_max[0],current_clock_counter,self.end_time))
    #                     if expired_qubit:
    #                         break
    #                     current_clock_counter-=self.time_granularity_value
    # #                     if global_printing_flag:
    # #                         print("an EPR was generated at time %s !"%(current_clock_counter))
    #                     if global_printing_flag:
    #                         print("........ we are updating all links")
    #                     for link in path_links:
    #                         for memory_id in range(self.each_path_memory_min[self.path_id],self.each_path_memory_max[self.path_id]):
    #                             left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[0],"right",memory_id]
    #                             right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[self.path_id,link[1],"left",memory_id]
    #                             expired_qubit = self.update(link,memory_id,current_clock_counter)
    #                             if expired_qubit:
    #                                 if global_printing_flag:
    #                                     print("qubit on link %s %s is expired!"%(link[0],link[1]))
    #                                 self.remove_all_generated_link_level_eprs(memory_id)
    #                                 break
    #                         if expired_qubit:
    #                             break
    #                     if expired_qubit:
    #                         break
    #                 if expired_qubit:
    #                     break
    #             if expired_qubit:
    #                 break
    #         if global_printing_flag:
    #             print("we will preform swaps ")
    #         for repeater in self.path_id_path_repeaters[self.path_id][1:-1]:
    #             self.swap_operation(current_clock_counter,repeater)
    #         if not expired_qubit:
    #             current_clock_counter+=self.time_granularity_value


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


# In[5]:


global_printing_flag = False
printing_qubit_aging_flag=False
printing_attempt_flag = False
printing_qubit_expiration_flag = False
printing_attempt_result_flag = False
printing_attempt_success_flag = False
global_qubit_expired_printing_flag = False
global_go_to_sleep_flag =False

global_number_of_sleeping_sec_attempt = 0.1
global_number_of_sleeping_sec_attempt_full_fail = 0.1
global_number_of_sleeping_sec_attempt_full_suc = 3
global_number_of_sleeping_sec_attempt_half_suc = 3
global_number_of_sleeping_sec_expiration = 6
global_number_of_sleeping_sec_swap = 2
global_number_of_sleeping_sec_delivered = 10

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
path_id_path_links = {0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11)]}


time_granularity_value = 3
experiment_name = "random_placement"
experiment_name="repeater_position"
# experiment_name="equal"
L0_list = np.linspace(10,800,101)
L0_list = [250]

# results_file_path = "results/fixed_distance_random_repeater_placementv2.csv"
results_file_path = "results/one_repeater_plaement_adjusting_exp_testing.csv"
results_file_path = "results/one_repeater_plaement_all_metrics_timestep_3.csv"
# results_file_path = "results/7_repeaters_equal_distance_no_cutoff.csv"
for iteration in range(100):
    for number_of_repeaters in [1]:    # for number_of_repeaters in [1]:
        path_id_path_repeaters = each_R_path_repeaters[number_of_repeaters]
        path_id_path_links =each_R_path_links[number_of_repeaters] 
        each_path_source_destination = {0:[0,number_of_repeaters+1]}
        initial_fidelity  = 1
        experimenting_classical_communication = True
        entanglement_generation_delay = True
        having_cut_offs = True
        considering_end_nodes_idle_time = False
        # for memory_max in [1,2,4,6,8,10,14,16]:
        #     for cut_off in [50,60,70,80,90,100,110,120,130,140,150,200,300]:
        #         for left_link_length in [4,6,8,10,12,14,16,18,20]:
        for memory_max in [1]:
            
            for i, L0 in enumerate(L0_list):
                each_link_length,rep_loc = compute_edges(experiment_name,L0,number_of_repeaters)
                print("each_link_length,rep_loc",each_link_length,rep_loc)
                import pdb
                # pdb.set_trace()
                for j, pos in enumerate(rep_loc):
                    if pos>=0.4:
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
                        running_time = 100000000
                        scheme = "sequential"
                        for path_id in path_id_path_links:
                            if not having_cut_offs:
                                system = System(scheme,path_id,running_time,1,1000,each_link_length,
                                                each_path_memory_min,each_path_memory_max)
                                system.main()
                                e2e_rate = system.e2e_EPRs/running_time*1000000
                                if e2e_rate>0:
                                    S,r,avg_e2e_f,e_x,e_z,e2e_fidelity= system.sekret_key(path_id,number_of_repeaters,e2e_rate)
                                else:
                                    S,r,avg_e2e_f,e_x,e_z,e2e_fidelity = 0,0,0,0,0,0
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
                                    
                                print("scheme %s L %s cut_off %s M %s time %s delivered %s e2e EPRs and e2e_rate %s SKR %s avg_e2e_f %s"%(scheme,L0,
                                                                                                                                            "no_cut_off",memory_max,
                                                                                                                                            running_time,system.e2e_EPRs,
                                                                                                                                            e2e_rate,S,avg_e2e_f))
                                if system.one_successful_round_times:
                                    avg_success_T = sum(system.one_successful_round_times)/len(system.one_successful_round_times)
                                else:
                                    avg_success_T = 0
                                Ns = len(system.one_successful_round_times)
                                Nf=len(system.one_failed_round_times)
                                if system.one_failed_round_times:
                                    avg_fail_T = sum(system.one_failed_round_times)/len(system.one_failed_round_times)
                                else:
                                    avg_fail_T = 0
                                with open(results_file_path, 'a') as newFile:                                
                                    newFileWriter = csv.writer(newFile)
                                    newFileWriter.writerow([scheme,experimenting_classical_communication,
                                                            having_cut_offs,
                                                            number_of_repeaters,
                                                            0,memory_max,running_time,
                                                            e2e_rate,system.longest_link_duration,
                                                           entanglement_generation_delay,path_id,
                                                           S,r,avg_e2e_f,L0,avg_T,e_x,e_z,
                                             system.expired_qubits_counter,
                                         rate_of_generating_first_segment,
                                             number_of_successful,experiment_name,iteration,i,j,having_cut_offs,e2e_fidelity,
                                                            Ns,avg_success_T,Nf,avg_fail_T])
                            else:
                                # for cut_off in [500, 1000, 2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000, 60000,80000,100000,140000,200000]:
                                # for cut_off in range(1,100,5):
                                for cut_off in [50]:
                                    cut_off = cut_off*1000
                                    system = System(scheme,path_id,running_time,1,cut_off,each_link_length,
                                                    each_path_memory_min,each_path_memory_max)
                                    system.main()
                                    e2e_rate = system.e2e_EPRs/running_time*1000000
                                    if e2e_rate>0:
                                        S,r,avg_e2e_f,e_x,e_z,e2e_fidelity= system.sekret_key(path_id,number_of_repeaters,e2e_rate)
                                    else:
                                        S,r,avg_e2e_f,e_x,e_z,e2e_fidelity = 0,0,0,0,0,0
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
                                    print("scheme %s L %s cut_off %s M %s time %s delivered %s e2e EPRs and e2e_rate %s SKR %s avg_e2e_f %s"%(scheme,
                                                                                                                                            L0,
                                                                                                                                            cut_off,
                                                                                                                                            memory_max,
                                                                                                                                            running_time,
                                                                                                                                            system.e2e_EPRs,
                                                                                                                                            e2e_rate,S,
                                                                                                                                            avg_e2e_f))
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


                                    line_items = [scheme,experimenting_classical_communication,
                                                                having_cut_offs,
                                                                number_of_repeaters,
                                                                cut_off,memory_max,running_time,
                                                                e2e_rate,system.longest_link_duration,
                                                               entanglement_generation_delay,path_id,
                                                               S,r,avg_e2e_f,L0,avg_T,e_x,e_z,
                                                 system.expired_qubits_counter,
                                             rate_of_generating_first_segment,
                                                 number_of_successful,experiment_name,iteration,i , j,having_cut_offs,e2e_fidelity]
                                    line_items.append(Ns)
                                    line_items.append(avg_success_T)
                                    line_items.append(Nf)
                                    line_items.append(avg_fail_T)
                                    # print("we done with L %s "%(L0))
                                    # time.sleep(20)
                                    with open(results_file_path, 'a') as newFile:                                
                                        newFileWriter = csv.writer(newFile)
                                        newFileWriter.writerow([item for item in line_items])
                                    print("we appended!")
                                    # time.sleep(100)
                


# In[ ]:


# exp = np.exp(-126)
# print(1/2+1/2*(product)*(exp))


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
#             having_cut_offs = False
#             results_file_path = "results/asynch_repeater_chain_results_repeater_position_expv3.csv"
#             # for memory_max in [1,2,4,6,8,10,14,16]:
#             #     for cut_off in [50,60,70,80,90,100,110,120,130,140,150,200,300]:
#             #         for left_link_length in [4,6,8,10,12,14,16,18,20]:
#             for memory_max in [1]:
#         #         for left_link_length in [1,50,100,5,10,20,30,40,50,15,25,35,45,55,60,65,70,75,80,90,200,300,400,500,600,700,800]:
#         #         for left_link_length in [10]:



# #                 L0 = 100 # e2e distance [km]
#                 c = 2e5 # speed of light [km/s]
#                 τ_coh = 0.1

#                 δ = 0.1
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
#                     scheme = "sequential"
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
            


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




