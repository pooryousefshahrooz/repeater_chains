#!/usr/bin/env python
# coding: utf-8

# In[3]:


import random
import time
import csv
import networkx as nx


# In[61]:


class Memory:
    def __init__(self,mode,repeater_id,entangled_node,memory_id,longest_link_duration,
                 attached_link_duration,initial_fidelity,
                 EPR_gen_success_p,cut_off):
        self.mode = mode
        self.entangled_node = entangled_node
        self.entangled_memory_id = memory_id
        self.EPR_age = -1 # we track the age of the EPR pair generated for this memory
        self.memory_id = memory_id
        self.repeater_id = repeater_id
        self.attempt_flag = False# check if we have started the attempt 
        self.synchronous_flag = False # indicates you can not try even if you have failed untile next time slot
        self.attempt_start_time = 0 # we track when we have started the attempt
        self.attempt_finishing_time = 0 # when would we get the ack for the attempt
        self.initial_fidelity = initial_fidelity# track the fidleity of the EPR pair stored on this memory
        self.attached_link_duration = attached_link_duration # how long it takes to send photon one way
        self.attempt_success_p = EPR_gen_success_p
        self.cut_off = cut_off
        self.longest_link_duration = longest_link_duration
        self.each_time_slot_duration = 3*longest_link_duration
        
        
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
        
        if global_printing_flag:
            print("repeater %s sends a %s message which will arrive at %s and now is %s"%(repeater,message_type,
                                                                                          self.arriving_time,
                                                                                          clock_counter))
        
        
class System:
    def __init__(self,mode,end_time,q_value,cut_off,each_link_length,each_path_memory_min,each_path_memory_max):
        self.mode = mode
        self.end_time = end_time
        self.experimenting_classical_communication = experimenting_classical_communication
        self.having_cut_offs = having_cut_offs
        self.entanglement_generation_delay = entanglement_generation_delay
        self.q_value = q_value
        self.e2e_EPRs = 0
        self.clock_counter = 0
        self.synchronous_flag = False
        self.initial_fidelity = initial_fidelity 
        self.cut_off= cut_off
        self.path_id_path_repeaters =path_id_path_repeaters
        self.path_id_path_links = path_id_path_links
        self.each_path_source_destination = each_path_source_destination
        self.each_link_cut_off  ={}
        self.each_link_length = each_link_length
        self.G = nx.Graph()
        for path, edges in self.path_id_path_links.items():
            for edge in edges:
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
        for path_id in self.path_id_path_links:
            self.each_path_swap_on_source_node_qubit_arrived[path_id]=False
            self.each_path_swap_on_source_node_qubit_id[path_id] = -1
            self.each_path_swap_on_end_node_qubit_id[path_id] = -1
            for repeater in self.path_id_path_repeaters[path_id][1:-1]:
                self.swap_list[repeater]=[]
                self.swap_start_sending_time_list[repeater] = []   

            for repeater in self.path_id_path_repeaters[path_id]:
                for memory_id in range(self.each_path_memory_min[path_id],self.each_path_memory_max[path_id]):
                    if repeater == self.each_path_source_destination[path_id][0]:
                        self.each_repeater_left_right_memory_exist_flag[path_id,repeater,"right",memory_id] =False
                    elif repeater == self.each_path_source_destination[path_id][1]:
                        self.each_repeater_left_right_memory_exist_flag[path_id,repeater,"left",memory_id] =False
                    else:
                        self.each_repeater_left_right_memory_exist_flag[path_id,repeater,"right",memory_id] =False
                        self.each_repeater_left_right_memory_exist_flag[path_id,repeater,"left",memory_id] =False
        #if self.mode=="synch1" or self.mode=="synch2":
        link_lengths = []
        for path_id,path_links in self.path_id_path_links.items():
            repeater_index = 0
            for link in path_links:
                length = self.each_link_length[link]
                link_lengths.append(length)
                repeater_index+=1
        longest_link = max(link_lengths)
        self.longest_link_duration = int((1.44*longest_link)/ 299792*1000000)
            
        for path_id,path_links in self.path_id_path_links.items():
            for link in path_links:
                for memory_id in range(self.each_path_memory_min[path_id],self.each_path_memory_max[path_id]):
                    length = self.each_link_length[link]
                    link_success_p = 10**(-0.2*length/10)
#                     link_success_p = 1
                    link_duration = int((1.44*length)/ 299792*1000000)
                    left_memory_object_instanse = Memory(self.mode,link[0],link[1],memory_id,self.longest_link_duration,
                                           link_duration,self.initial_fidelity,
                                           link_success_p,self.cut_off)
                    right_memory_object_instanse = Memory(self.mode,link[1],link[0],memory_id,self.longest_link_duration,
                                           link_duration,self.initial_fidelity,
                                           link_success_p,self.cut_off)
                    self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[0],"right",memory_id] = left_memory_object_instanse
                    self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[1],"left",memory_id] = right_memory_object_instanse
                    

    
  
    
    def shortest_path_to_end_nodes_length(self,repeater,path_id):
        source = self.each_path_source_destination[path_id][0]
        destination = self.each_path_source_destination[path_id][1]
        shortest_path_length1  = nx.shortest_path_length(self.G, source=repeater, target=source,weight = "weight")
        shortest_path_length2  = nx.shortest_path_length(self.G, source=repeater, target=destination,weight = "weight")
        return shortest_path_length1,source
    
        
                        
                        
    
                    


    def copy_left_instanse_to_right_instance(self,left_memory_object,right_memory_object):
        right_memory_object.attempt_flag = left_memory_object.attempt_flag
        right_memory_object.synchronous_flag = left_memory_object.synchronous_flag
        right_memory_object.attempt_start_time = left_memory_object.attempt_start_time
        right_memory_object.attempt_finishing_time = left_memory_object.attempt_finishing_time
        right_memory_object.initial_fidelity = left_memory_object.initial_fidelity
        right_memory_object.longest_link_duration = left_memory_object.longest_link_duration
        right_memory_object.each_time_slot_duration = left_memory_object.each_time_slot_duration
        
    def update(self,path_id,link,memory_id,current_clock_counter):
#         print("going to attempt!")
        expired_flag = False
        left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[0],"right",memory_id]
        right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[1],"left",memory_id]
        left_node_memory_exist_flag = self.each_repeater_left_right_memory_exist_flag[path_id,link[0],"right",memory_id]
        right_node_memory_exist_flag = self.each_repeater_left_right_memory_exist_flag[path_id,link[1],"left",memory_id]
#         if global_printing_flag:
#             print("attemptig generating EPR at %s on link (%s,%s) flags %s %s "%(current_clock_counter,
#                                                         left_memory_object.repeater_id,
#                                                         left_memory_object.entangled_node,
#                                                         left_node_memory_exist_flag,
#                                                         right_node_memory_exist_flag
#                                                                                                      ))
        
        if (not left_memory_object.attempt_flag and left_node_memory_exist_flag) or (not left_memory_object.attempt_flag and right_node_memory_exist_flag):
#             print("we have not tried to generate but we have swapped on left and right")
            left_memory_object.EPR_age+=1
            right_memory_object.EPR_age+=1
#             print("1 link (%s,%s) got aged by one new age %s "%(link[0],link[1],left_memory_object.EPR_age))
            if left_memory_object.EPR_age >self.each_link_cut_off[link] and self.having_cut_offs:
                if global_printing_flag:
                    print("0!!!!!unfortunately the qubit got aged at link %s %s  age %s cut_off %s at time %s "%(left_memory_object.repeater_id,left_memory_object.entangled_node,left_memory_object.EPR_age, self.cut_off,current_clock_counter))
                self.send_a_message(link[0],path_id,"cancelling",left_memory_object.EPR_age,
                                                        right_memory_object.EPR_age,
                                                        link[0],
                                                        link[1],
                                                        left_memory_object.memory_id,
                                                        right_memory_object.memory_id,
                                                        current_clock_counter,False)
                
                left_memory_object.attempt_flag = False
                left_memory_object.EPR_age = -1
                expired_flag= True
                """we send a message to the source node to execute these commands locally there. 
                We need to have a local version of this for links to know when they can attempt"""
                self.each_repeater_left_right_memory_exist_flag[path_id,link[0],"right",memory_id] = False
                self.each_repeater_left_right_memory_exist_flag[path_id,link[1],"left",memory_id]  = False
            
        elif left_memory_object.attempt_flag and not left_node_memory_exist_flag and not right_node_memory_exist_flag:#we have attempted before lets check if it is done
            left_memory_object.EPR_age+=1
            right_memory_object.EPR_age+=1
#             print("2 link (%s,%s) got aged by one new age %s "%(link[0],link[1],left_memory_object.EPR_age))
            if left_memory_object.EPR_age >self.each_link_cut_off[link] and self.having_cut_offs:
                if global_printing_flag:
                    print("1!!!!!! unfortunately the qubit got aged at link %s %s  age %s cut_off %s at time %s "%(left_memory_object.repeater_id,left_memory_object.entangled_node,left_memory_object.EPR_age, self.cut_off,current_clock_counter))
                expired_flag =True
                self.send_a_message(link[0],path_id,"cancelling",left_memory_object.EPR_age,
                                                        right_memory_object.EPR_age,
                                                        link[0],
                                                        link[1],
                                                        left_memory_object.memory_id,
                                                        right_memory_object.memory_id,
                                                        current_clock_counter,False)
                
                left_memory_object.attempt_flag = False
                left_memory_object.EPR_age = -1
                self.each_repeater_left_right_memory_exist_flag[path_id,link[0],"right",memory_id] = False
                self.each_repeater_left_right_memory_exist_flag[path_id,link[1],"left",memory_id]  = False
            
                    
                                        
        self.copy_left_instanse_to_right_instance(left_memory_object,right_memory_object)
                    
            
        return expired_flag
            
    def attempt_to_generate(self,path_id,link,memory_id,current_clock_counter):
#         print("going to attempt!")
        left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[0],"right",memory_id]
        right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[1],"left",memory_id]
        left_node_memory_exist_flag = self.each_repeater_left_right_memory_exist_flag[path_id,link[0],"right",memory_id]
        right_node_memory_exist_flag = self.each_repeater_left_right_memory_exist_flag[path_id,link[1],"left",memory_id]
        generated_flag = False
#         if global_printing_flag:
#             print("attemptig generating EPR at %s on link (%s,%s) flags %s %s "%(current_clock_counter,
#                                                         left_memory_object.repeater_id,
#                                                         left_memory_object.entangled_node,
#                                                         left_node_memory_exist_flag,
#                                                         right_node_memory_exist_flag
#                                                                                                      ))
        if (not left_memory_object.attempt_flag and 
            not left_node_memory_exist_flag and 
            not right_node_memory_exist_flag) or (left_memory_object.mode =="synch1" and self.failed_at_least_one_link):
#             if left_memory_object.synchronous_flag and left_memory_object.mode =="synch1":
#                 pass
#             else:
                left_memory_object.attempt_flag = True
                left_memory_object.entangled_node = link[1]
                right_memory_object.entangled_node = link[0]
                left_memory_object.attempt_start_time = current_clock_counter
                left_memory_object.EPR_age = 0
                self.each_repeater_left_right_memory_exist_flag[path_id,link[0],"right",memory_id] =False
                self.each_repeater_left_right_memory_exist_flag[path_id,link[1],"left",memory_id] =False
                if left_memory_object.mode =="synch1" or left_memory_object.mode =="synch2":
                    if self.experimenting_classical_communication:
                        left_memory_object.attempt_finishing_time = current_clock_counter+2*left_memory_object.longest_link_duration
                    else:
                        left_memory_object.attempt_finishing_time = current_clock_counter+1
                else:
                    if self.experimenting_classical_communication:
                        left_memory_object.attempt_finishing_time = current_clock_counter+2*left_memory_object.attached_link_duration
                    else:
                        left_memory_object.attempt_finishing_time = current_clock_counter+1
                if global_printing_flag:
                    print("statrt attemptig to generate EPR at %s on link (%s,%s) that will finish at %s "%(current_clock_counter,
                                                                                                      left_memory_object.repeater_id,
                                                                                                      left_memory_object.entangled_node,
                                                                                                     left_memory_object.attempt_finishing_time))
                    
                    
                    
                    
        elif (not left_memory_object.attempt_flag and left_node_memory_exist_flag) or (not left_memory_object.attempt_flag and right_node_memory_exist_flag):
#             print("we have not tried to generate but we have swapped on left and right")
            left_memory_object.EPR_age+=1
            right_memory_object.EPR_age+=1
            if left_memory_object.EPR_age >self.each_link_cut_off[link] and self.having_cut_offs:
                if global_printing_flag:
                    print("0!!!!!unfortunately the qubit got aged at link %s %s  age %s cut_off %s at time %s "%(left_memory_object.repeater_id,left_memory_object.entangled_node,left_memory_object.EPR_age, self.cut_off,current_clock_counter))
                self.send_a_message(link[0],path_id,"cancelling",left_memory_object.EPR_age,
                                                        right_memory_object.EPR_age,
                                                        link[0],
                                                        link[1],
                                                        left_memory_object.memory_id,
                                                        right_memory_object.memory_id,
                                                        current_clock_counter,False)
                
                left_memory_object.attempt_flag = False
                left_memory_object.EPR_age = -1
                
                """we send a message to the source node to execute these commands locally there. 
                We need to have a local version of this for links to know when they can attempt"""
                self.each_repeater_left_right_memory_exist_flag[path_id,link[0],"right",memory_id] = False
                self.each_repeater_left_right_memory_exist_flag[path_id,link[1],"left",memory_id]  = False
            
        elif left_memory_object.attempt_flag and not left_node_memory_exist_flag and not right_node_memory_exist_flag:#we have attempted before lets check if it is done
            left_memory_object.EPR_age+=1
            if left_memory_object.EPR_age >self.each_link_cut_off[link] and self.having_cut_offs:
                if global_printing_flag:
                    print("1!!!!!! unfortunately the qubit got aged at link %s %s  age %s cut_off %s at time %s "%(left_memory_object.repeater_id,left_memory_object.entangled_node,left_memory_object.EPR_age, self.cut_off,current_clock_counter))
                
                self.send_a_message(link[0],path_id,"cancelling",left_memory_object.EPR_age,
                                                        right_memory_object.EPR_age,
                                                        link[0],
                                                        link[1],
                                                        left_memory_object.memory_id,
                                                        right_memory_object.memory_id,
                                                        current_clock_counter,False)
                
                left_memory_object.attempt_flag = False
                left_memory_object.EPR_age = -1
                self.each_repeater_left_right_memory_exist_flag[path_id,link[0],"right",memory_id] = False
                self.each_repeater_left_right_memory_exist_flag[path_id,link[1],"left",memory_id]  = False
            else:
                if left_memory_object.attempt_finishing_time == current_clock_counter:
                    random_value = random.uniform(0,1)
                    if random_value<= left_memory_object.attempt_success_p:
                        if global_printing_flag:
                            print("EPR pair on link %s,%s was generated at time %s! "%(left_memory_object.repeater_id,left_memory_object.entangled_node,current_clock_counter))
                        left_memory_object.attempt_flag = False
                        generated_flag = True
                        if left_memory_object.mode =="synch1" or left_memory_object.mode =="synch2":
                            left_memory_object.EPR_age = 2*left_memory_object.longest_link_duration
                            right_memory_object.EPR_age = 1*left_memory_object.longest_link_duration
                        else:
                            if self.entanglement_generation_delay:
                                left_memory_object.EPR_age = 2*left_memory_object.attached_link_duration
                                right_memory_object.EPR_age = 1*left_memory_object.attached_link_duration
                            else:
                                left_memory_object.EPR_age = 0
                                right_memory_object.EPR_age = 0
                        left_memory_object.fidelity = self.initial_fidelity
                        self.each_repeater_left_right_memory_exist_flag[path_id,link[0],"right",memory_id] = True
                        self.each_repeater_left_right_memory_exist_flag[path_id,link[1],"left",memory_id]  = True
                    else:
                        if global_printing_flag:
                            print("EPR pair on link %s,%s was failed at time %s ! "%(left_memory_object.repeater_id,left_memory_object.entangled_node,current_clock_counter))
                        if left_memory_object.mode=="synch1":
                            self.synchronous_flag = True
                        generated_flag = False
                        self.failed_at_least_one_link = True
                        left_memory_object.attempt_flag = False
                        left_memory_object.EPR_age = -1

                    
                                        
        self.copy_left_instanse_to_right_instance(left_memory_object,right_memory_object)
                
        return generated_flag
            
    def remove_all_generated_link_level_eprs(self,path_id,memory_id):
        global global_printing_flag
        if global_printing_flag:
            print("we are removing whatever link that has been generated so far")
        
        
        for link in self.path_id_path_links[path_id]:
            left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[0],"right",memory_id]
            right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[1],"left",memory_id]
            left_memory_object.attempt_flag = False
            left_memory_object.EPR_age = -1
            self.each_repeater_left_right_memory_exist_flag[path_id,link[0],"right",memory_id] = False
            self.each_repeater_left_right_memory_exist_flag[path_id,link[1],"left",memory_id]  = False
            self.copy_left_instanse_to_right_instance(left_memory_object,right_memory_object)
    def send_a_message(self,repeater,path_id,message_type,left_node_qubit_age,
                       right_node_qubit_age,left_node,right_node,
                       memory_id_left,memory_id_right,
                       current_clock_counter,success_flag):
        length,destination = self.shortest_path_to_end_nodes_length(repeater,path_id)
        link_duration = int((1.44*length)/ 299792*1000000)
        
        message = Message(self.experimenting_classical_communication,repeater,destination,path_id,
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
        if global_printing_flag:
            print("swap result arrived from %s at %s on qubits with ages %s %s !rec./unrec. swap results: %s "%(message.sender,clock_counter,message.left_qubit_age,message.right_qubit_age,self.swap_list))
        if message.left_node==self.each_path_source_destination[message.path_id][0]:
#             print("************************************ this arrived swap is entangeled with the source node")
            self.each_path_swap_on_source_node_qubit_arrived[message.path_id] = True
            self.each_path_swap_on_source_node_qubit_id[message.path_id] = message.memory_id_right
        if message.right_node==self.each_path_source_destination[message.path_id][1]:
            self.each_path_swap_on_end_node_qubit_id[message.path_id]=message.memory_id_right

    def check_swap_results_arrived(self,path_id,clock_counter):
#         print("we are checking if there is any arrived message")
        for message in self.message_channel:
#             print("for message from ",message.sender,message.arriving_time,clock_counter)
            if message.arriving_time==clock_counter:
                if message.type =="cancelling":
                    self.receive_a_fizzling_message(message,clock_counter)
                elif message.type =="swap_result":
                    self.receive_a_swap_result_message(message,clock_counter)
                self.message_channel.remove(message)
        if self.each_path_swap_on_source_node_qubit_arrived[path_id]:
            missing_one_swap = False
            for repeater in self.path_id_path_repeaters[path_id][1:-1]:
                if len(self.swap_list[repeater])>0:
                    if self.swap_list[repeater][0]==0:
                        missing_one_swap= True
                else:
                    missing_one_swap= True
            if not missing_one_swap:
#                 print("**************************************************** we release the source node qubit at time %s by settign it to False "%(clock_counter))
                self.each_repeater_left_right_memory_exist_flag[path_id,self.each_path_source_destination[path_id][0],"right",self.each_path_swap_on_source_node_qubit_id[path_id]] =False
                self.each_repeater_left_right_memory_exist_flag[path_id,self.each_path_source_destination[path_id][1],"left",self.each_path_swap_on_end_node_qubit_id[path_id]] =False

    def check_all_swaps(self,path_id,clock_counter):
        missing_one_swap = False
        for repeater in self.path_id_path_repeaters[path_id][1:-1]:
            if len(self.swap_list[repeater])>0:
                if self.swap_list[repeater][0]==0:
                    missing_one_swap= True
            else:
                missing_one_swap= True
                
        if not missing_one_swap:
            if global_printing_flag:
                print("********************* we delivered one e2e EPR pair at time %s *********messages in channel %s***************"%(clock_counter,len(self.message_channel)))
            self.e2e_EPRs+=1
            for repeater in self.path_id_path_repeaters[path_id][1:-1]: 
                if len(self.swap_list[repeater])==1:
                    self.swap_list[repeater] =[]
                    self.swap_start_sending_time_list[repeater] = []
                elif len(self.swap_list[repeater])>1:
                    self.swap_list[repeater] = self.swap_list[repeater][1:]
                    self.swap_start_sending_time_list[repeater] = self.swap_start_sending_time_list[repeater][1:]
            self.each_path_swap_on_source_node_qubit_arrived[path_id]=False
            self.each_path_swap_on_source_node_qubit_id[path_id] = -1
            self.each_path_swap_on_end_node_qubit_id[path_id] = -1
            
    def swap_operation(self,current_clock_counter,path_id,repeater):
        global global_printing_flag
        left_side_memory_ages = []
        right_side_memory_ages = []
        left_memory_objects = []
        right_memory_objects = []
        for memory_id in range(self.each_path_memory_min[path_id],self.each_path_memory_max[path_id]):
            left_side_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,repeater,"left",memory_id]
            right_side_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,repeater,"right",memory_id]
            node_left_memory_id_exist_flag = self.each_repeater_left_right_memory_exist_flag[path_id,repeater,"left",memory_id]
            node_right_memory_id_exist_flag = self.each_repeater_left_right_memory_exist_flag[path_id,repeater,"right",memory_id]
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
                                self.each_repeater_left_right_memory_exist_flag[path_id,repeater,"right",right_memory_object.memory_id] =False
                            
                                self.each_repeater_left_right_memory_exist_flag[path_id,repeater,"left",left_memory_object.memory_id] =False
                                self.failed_at_least_one_link = False
                                if random_value <=self.q_value:
                                    
                                    left_entangled_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,left_memory_object.entangled_node,"right",left_memory_object.entangled_memory_id]
                                    right_entangled_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,right_memory_object.entangled_node,"left",right_memory_object.entangled_memory_id]

                                    
                                    self.send_a_message(repeater,path_id,"swap_result",left_memory_object.EPR_age,
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
                                    self.each_repeater_left_right_memory_exist_flag[path_id,repeater,"right",right_memory_object.memory_id] =False
                                    self.each_repeater_left_right_memory_exist_flag[path_id,repeater,"left",left_memory_object.memory_id] =False
                                    self.each_repeater_left_right_memory_exist_flag[path_id,left_memory_object.entangled_node,"right",left_memory_object.memory_id] =False
                                    self.each_repeater_left_right_memory_exist_flag[path_id,right_memory_object.entangled_node,"left",right_memory_object.memory_id] =False
                                    left_entangled_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,left_memory_object.entangled_node,"right",left_memory_object.entangled_memory_id]
                                    right_entangled_node_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,right_memory_object.entangled_node,"left",right_memory_object.entangled_memory_id]

                                    left_entangled_node_memory_object.attempt_flag = False
                                    right_entangled_node_memory_object.attempt_flag = False
                                    left_memory_object.attempt_flag = False
                                    right_memory_object.attempt_flag = False                    
                                    left_memory_object.EPR_age = -1
                                    right_memory_object.EPR_age = -1
        
        
        
    def main(self):
        self.e2e_EPRs = 0
        current_clock_counter = 0
        while(current_clock_counter <= self.end_time):
            
            if current_clock_counter%5000==0:
                print("mode %s cut_off %s link %s M %s clock %s End %s "%(self.mode,self.cut_off,self.each_link_length[(0,1)],self.each_path_memory_max[0],current_clock_counter,self.end_time))
            for path_id,path_links in self.path_id_path_links.items():
                for link in path_links:
                    if global_printing_flag:
                        print("time %s for link %s "%(current_clock_counter,link))
                    for memory_id in range(self.each_path_memory_min[path_id],self.each_path_memory_max[path_id]):
                        left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[0],"right",memory_id]
                        right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[1],"left",memory_id]
                        generated_a_link = False
                        while(current_clock_counter<= self.end_time and not generated_a_link):
                            generated_a_link = self.attempt_to_generate(path_id,link,memory_id,current_clock_counter)
                            self.check_swap_results_arrived(path_id,current_clock_counter)
                            self.check_all_swaps(path_id,current_clock_counter)
                            current_clock_counter+=1
                        current_clock_counter-=1
                        if global_printing_flag:
                            print("an EPR was generated at time %s !"%(current_clock_counter))
                        expired_qubit = False
                        for link in path_links:
                            for memory_id in range(self.each_path_memory_min[path_id],self.each_path_memory_max[path_id]):
                                left_memory_object  = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[0],"right",memory_id]
                                right_memory_object = self.each_path_repeater_left_right_memory_id_memory_object[path_id,link[1],"left",memory_id]
                                
                                expired_qubit = self.update(path_id,link,memory_id,current_clock_counter)
                                if expired_qubit:
                                    if global_printing_flag:
                                        print("qubit on link %s %s is expired!"%(link[0],link[1]))
                                    self.remove_all_generated_link_level_eprs(path_id,memory_id)
                                    break
                            if expired_qubit:
                                break
                        if expired_qubit:
                            break
                    if expired_qubit:
                        break
                if global_printing_flag:
                    print("we will preform swaps ")
                for repeater in self.path_id_path_repeaters[path_id][1:-1]:
                    self.swap_operation(current_clock_counter,path_id,repeater)
            current_clock_counter+=1


# In[ ]:





# In[62]:


global_printing_flag = False

each_R_path_repeaters = {1:{0:[0,1,2]},
                         2:{0:[0,1,2,3]},
                         3:{0:[0,1,2,3,4]},
                         4:{0:[0,1,2,3,4,5]},
                         5:{0:[0,1,2,3,4,5,6]},
                         6:{0:[0,1,2,3,4,5,6,7]}
                        }

each_R_path_links = {    1:{0:[(0,1),(1,2)]},
                         2:{0:[(0,1),(1,2),(2,3)]},
                         3:{0:[(0,1),(1,2),(2,3),(3,4)]},
                         4:{0:[(0,1),(1,2),(2,3),(3,4),(4,5)]},
                         5:{0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]},
                         6:{0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]}
                        }
path_id_path_repeaters = {0:[0,1,2,3,4,5,6,7,8]}
path_id_path_links = {0:[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)]}

for number_of_repeaters in [1,2,3,4,5,6]:
    path_id_path_repeaters = each_R_path_repeaters[number_of_repeaters]
    path_id_path_links =each_R_path_links[number_of_repeaters] 
    each_path_source_destination = {0:[0,number_of_repeaters+1]}
    initial_fidelity  = 0.95
    experimenting_classical_communication = False
    entanglement_generation_delay = False
    having_cut_offs = False
    results_file_path = "results/sequential_repeater_chain_results.csv"
    # for memory_max in [1,2,4,6,8,10,14,16]:
    #     for cut_off in [50,60,70,80,90,100,110,120,130,140,150,200,300]:
    #         for left_link_length in [4,6,8,10,12,14,16,18,20]:
    for memory_max in [1]:
        for left_link_length in [1,50,100,200,300,400,500,600,700,800,5,10,20,30,40,50,15,25,35,45]:
#         for left_link_length in [10]:
            each_link_length = {(0,1):left_link_length,(1,2):left_link_length,(2,3):left_link_length,
                                (3,4):left_link_length,
                               (4,5):left_link_length,
                               (5,6):left_link_length,
                               (6,7):left_link_length,
                               (7,8):left_link_length}
            each_path_memory_min = {0:0}
            each_path_memory_max= {0:memory_max}
            running_time = 4000000
            scheme = "sequential"
            if not having_cut_offs:
                system = System(scheme,running_time,1,1000,each_link_length,
                                each_path_memory_min,each_path_memory_max)
                system.main()
                e2e_rate = system.e2e_EPRs/running_time
                print("*****!!!!**** scheme %s L %s cut_off %s M %s time %s delivered %s e2e EPRs and e2e_rate %s"%(scheme,left_link_length,"no_cut_off",memory_max,running_time,system.e2e_EPRs,e2e_rate))
                with open(results_file_path, 'a') as newFile:                                
                    newFileWriter = csv.writer(newFile)
                    newFileWriter.writerow([scheme,experimenting_classical_communication,
                                            having_cut_offs,
                                            number_of_repeaters,
                                            left_link_length,False,memory_max,running_time,
                                            e2e_rate,system.longest_link_duration,
                                           entanglement_generation_delay])
            else:
                for cut_off in [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000, 30000, 40000, 60000]:
#                 for cut_off in [100,500,10000]:
                    system = System(scheme,running_time,1,cut_off,each_link_length,
                                    each_path_memory_min,each_path_memory_max)
                    system.main()
                    e2e_rate = system.e2e_EPRs/running_time
                    print("*****!!!!**** scheme %s L %s cut_off %s M %s time %s delivered %s e2e EPRs and e2e_rate %s"%(scheme,left_link_length,cut_off,memory_max,running_time,system.e2e_EPRs,e2e_rate))
                    with open(results_file_path, 'a') as newFile:                                
                        newFileWriter = csv.writer(newFile)
                        newFileWriter.writerow([scheme,experimenting_classical_communication,
                                                having_cut_offs,
                                                number_of_repeaters,
                                                left_link_length,cut_off,memory_max,running_time,
                                                e2e_rate,system.longest_link_duration,
                                               entanglement_generation_delay])
            


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




