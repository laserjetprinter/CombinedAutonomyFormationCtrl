########################################
# File: slot_wb.py
# Author: Natalie Davis - 4/22/2024
# Description: Weight-based and slot offset formation classes. Agents can be
# flown in individal formations or combined. The Container class provides
# simulation stepping and visualization. Once done with simulation visualization,
# exit out of the plot to view the individual agent formation position error
# over dt.
########################################

import numpy as np
import math
import matplotlib.pyplot as plt
import random
plt.rcParams["figure.figsize"] = 7,7
from matplotlib.animation import FuncAnimation


global fig
global ax 

class WB_Agent:
    def __init__(self, comms_radius, edge_length, max_follower_speed, max_leader_speed, is_leader) -> None:
        ### Set up vars
        self.comms_radius = comms_radius
        self.edge_length = edge_length
        self.is_leader = is_leader
        self.max_follower_speed = max_follower_speed
        self.max_leader_speed = max_leader_speed
        self.agent_pos_error = []
        self.agent_dist_error = []
        self.agent_msgs = []

        if self.is_leader:
            self.max_speed = self.max_leader_speed
        else:
            self.max_speed = self.max_follower_speed

    def init_goal(self, start_pos, rendezvous_pos, trajectory_type, increment_val=5, r=40):
        if trajectory_type == "line":
            self.line_trajectory(start_pos, rendezvous_pos, increment_val)
        elif trajectory_type == "circle":
            self.circle_trajectory(rendezvous_pos, r)
        
    def circle_trajectory(self, rendezvous_pos, r): 
        self.goals = []
        self.goal_point = 0

        theta_values = np.linspace(0,2*np.pi,360)
        for theta in theta_values:
            self.goals.append(np.array([rendezvous_pos[0] + r*np.cos(theta), rendezvous_pos[1] + r*np.sin(theta)]))

        return self.goals

    def line_trajectory(self, start_pos, rendezvous_pos, increment_val):
        self.goals = []
        self.goal_point = 0

        x_values = np.linspace(start_pos[0],rendezvous_pos[0],increment_val)
        y_values = np.linspace(start_pos[1],rendezvous_pos[1],increment_val)
        for x_pos, y_pos in zip(x_values,y_values):
            self.goals.append(np.array([x_pos,y_pos]))

    def set_leader(self, is_leader):
        self.is_leader = is_leader

        ### Reset max speed
        if self.is_leader:
            self.max_speed = self.max_leader_speed
        else:
            self.max_speed = self.max_follower_speed

    def set_edge_length(self, edge_length):
        self.edge_length = edge_length
    
    def get_edge_length(self):
        return self.edge_length

    def create_msg(self, current_loc, agent_id) -> dict:
        self.current_location = current_loc
        self.agent_id = agent_id
        return {"current_loc": current_loc, "agent_id": agent_id}

    def append_msg(self, msg) -> None:
        self.agent_msgs.append(msg)

    def get_all_msgs(self):
        return self.agent_msgs

    def process_msgs(self) -> list:
        ij_prods = []
        x_sums = 0
        y_sums = 0

        if self.is_leader:
            goal_diff = abs(np.array(self.goals[self.goal_point]) - np.array(self.current_location))
            
            ### If at the goal, then increment to the next goal
            if goal_diff[0] < 2 and  goal_diff[1] < 2:
                if self.goal_point == (len(self.goals) - 1):
                    self.goal_point = 0
                else:
                    self.goal_point += 1

            ### Calculate the weights and vel for the next goal
            x_i = np.array(self.current_location)
            x_j = np.array(self.goals[self.goal_point])
            diff = np.linalg.norm(x_i - x_j) * (x_j - x_i)
            x_sums = diff[0]
            y_sums = diff[1]
        else:
            for msg in self.agent_msgs:
                x_i = np.array(self.current_location)
                x_j = np.array(msg["current_loc"])

                ### With comms__radius
                # This is not used in simulations with very large comms radii
                # p1 = 1 - (self.edge_length[msg["agent_id"]] / np.linalg.norm(x_i - x_j))
                # p2 = (comms_radius - np.linalg.norm(x_i - x_j))**3

                ### Without comms_radius
                p1 = np.linalg.norm(x_i - x_j) - self.edge_length[msg["agent_id"]]
                p2 = np.linalg.norm(x_i - x_j)

                w_ij = p1 / p2
                cur_weight_prod = w_ij * np.linalg.norm(x_i - x_j) * (x_j - x_i)
                ij_prods.append(cur_weight_prod)

            ### Clear agent messages after each time step
            self.agent_msgs = []

            ### Sum the ij products
            ij_array = np.array(ij_prods)
            x_sums = np.sum(ij_array[:,0])
            y_sums = np.sum(ij_array[:,1])

        ### Clip the speeds
        if x_sums > self.max_speed:
            x_sums = self.max_speed
        elif abs(x_sums) > self.max_speed:
            x_sums = -self.max_speed
        if y_sums > self.max_speed:
            y_sums = self.max_speed
        elif abs(y_sums) > self.max_speed:
            y_sums = -self.max_speed

        return (np.array([x_sums,y_sums]))

    def set_pos_error(self, pos_error):
        self.agent_pos_error.append(pos_error)

    def get_pos_error(self):
        return self.agent_pos_error

    def set_dist_error(self, dist_error):
        self.agent_dist_error.append(dist_error)
    
    def get_dist_error(self):
        return self.agent_dist_error

class Slot_Agent:
    def __init__(self, comms_radius, edge_lengths, slot_loc, follower_k, max_speed, is_leader) -> None:
        self.comms_radius = comms_radius
        self.edge_lengths = edge_lengths
        self.max_speed = max_speed
        self.is_leader = is_leader
        self.slot_loc = slot_loc
        self.follower_k = follower_k
        self.agent_pos_error = []
        self.agent_dist_error = []
        self.agent_msgs = []

    def init_goal(self, start_pos, rendezvous_pos, trajectory_type, increment_val=5, r=40):
        if trajectory_type == "line":
            self.line_trajectory(start_pos, rendezvous_pos, increment_val)
        elif trajectory_type == "circle":
            self.circle_trajectory(rendezvous_pos, r)

    def circle_trajectory(self, rendezvous_pos, r):
        self.goals = []
        self.goal_point = 0

        theta_values = np.linspace(0,2*np.pi,360)
        for theta in theta_values:
            self.goals.append(np.array([rendezvous_pos[0] + r*np.cos(theta), rendezvous_pos[1] + r*np.sin(theta)]))

        return self.goals
    
    def line_trajectory(self, start_pos, rendezvous_pos, increment_val):
        self.goals = []
        self.goal_point = 0
        
        x_values = np.linspace(start_pos[0],rendezvous_pos[0],increment_val)
        y_values = np.linspace(start_pos[1],rendezvous_pos[1],increment_val)
        for x_pos, y_pos in zip(x_values,y_values):
            self.goals.append(np.array([x_pos,y_pos]))

    def square_trajectory(self, map_limit):
        self.goals = []
        self.goal_point = 0

        x_values = np.linspace(-map_limit,map_limit,5)
        for point in x_values:
            self.goals.append(np.array([point,-map_limit]))

        y_values = np.linspace(-map_limit,map_limit,5)
        for point in y_values:
            self.goals.append(np.array([map_limit,point]))

        x_values = np.linspace(map_limit,-map_limit,5)
        for point in x_values:
            self.goals.append(np.array([point,map_limit]))

        y_values = np.linspace(map_limit,-map_limit,5)
        for point in y_values:
            self.goals.append(np.array([-map_limit,point]))

    def set_slot_loc(self,slot_loc):
        self.slot_loc = slot_loc

    def set_cur_loc(self,current_loc, agent_id):
        self.current_location = current_loc
        self.agent_id = agent_id

    def get_edge_length(self):
        return self.edge_lengths

    def create_leader_msg(self, current_loc, agent_id, leader_heading) -> dict:
        self.leader_location = current_loc
        self.leader_agent_id = agent_id
        self.leader_heading = leader_heading
        return {"current_loc": current_loc, "agent_id": agent_id, "leader_heading": leader_heading}

    def create_msg(self, current_loc, agent_id) -> dict:
        self.leader_location = current_loc
        self.leader_agent_id = agent_id
        return {"current_loc": current_loc, "agent_id": agent_id}

    def append_msg(self, msg) -> None:
        self.agent_msgs.append(msg)

    def get_all_msgs(self):
        return self.agent_msgs

    def calc_leader_heading(self):
        # Calculate the differences in x and y coordinates
        goal_diff = np.array(self.current_location) - np.array(self.goals[self.goal_point]) 
        
        # Calculate the heading (angle) using arctan2
        heading_rad = math.atan2(goal_diff[1], goal_diff[0])
        
        # Convert radians to degrees and normalize the angle to [0, 360)
        heading_deg = math.degrees(heading_rad) % 360

        return heading_deg

    def process_msgs(self) -> list:
        x_vel_val = 0
        y_vel_val = 0

        if self.is_leader:
            goal_diff = abs(np.array(self.goals[self.goal_point]) - np.array(self.current_location))

            # If at the goal, then increment to the next goal
            if goal_diff[0] < 2 and  goal_diff[1] < 2:
                if self.goal_point == (len(self.goals) - 1):
                    self.goal_point = 0
                else:
                    self.goal_point += 1

            # Calculate the weights and vel for the next goal
            x_i = np.array(self.current_location)
            x_j = np.array(self.goals[self.goal_point])
            diff = np.linalg.norm(x_i - x_j) * (x_j - x_i)
            x_vel_val = diff[0]
            y_vel_val = diff[1]

            ### Plot positional error calculation
            # For now having the leader have 0 error, because goal locations are always
            # projected in front of the agent
            pos_error = 0
            self.set_pos_error(pos_error)

        else:
            for msg in self.agent_msgs:
                x_i = np.array(self.current_location)
                leader_pos = np.array(msg["current_loc"])
                leader_heading_rads = math.radians(msg["leader_heading"] - 90) # Subtract 90 so that slots shift as if leader is at 0,0

                goal_x = leader_pos[0] - (math.cos(leader_heading_rads) * self.slot_loc[0] - math.sin(leader_heading_rads) * self.slot_loc[1])
                goal_y = leader_pos[1] - (math.sin(leader_heading_rads) * self.slot_loc[0] + math.cos(leader_heading_rads) * self.slot_loc[1])
                goal_loc = np.array((goal_x,goal_y))

                diff = goal_loc - x_i
                agent_vel = diff * self.follower_k

                ### Plot 
                # Positional error calculation
                pos_error = (diff[0]**2 + diff[1]**2)**(1/2)
                self.set_pos_error(pos_error)

            # Save the x and y components of the agent velocity
            x_vel_val = agent_vel[0]
            y_vel_val = agent_vel[1]

        # Clear the message queue after every step call
        self.agent_msgs = []

        # Clip the speeds
        if x_vel_val > self.max_speed:
            x_vel_val = self.max_speed
        elif abs(x_vel_val) > self.max_speed:
            x_vel_val = -self.max_speed
        if y_vel_val > self.max_speed:
            y_vel_val = self.max_speed
        elif abs(y_vel_val) > self.max_speed:
            y_vel_val = -self.max_speed

        return (np.array([x_vel_val,y_vel_val]))

    def set_pos_error(self, pos_error):
        self.agent_pos_error.append(pos_error)

    def get_pos_error(self):
        return self.agent_pos_error

    def set_dist_error(self, dist_error):
        self.agent_dist_error.append(dist_error)
    
    def get_dist_error(self):
        return self.agent_dist_error

class Container:
    wb_agents = []
    wb_agent_locs = []
    slot_agents = []
    slot_agent_locs = []

    def __init__(self, wb_init_locs, slot_init_locs, dt, ax, tracked_edges, graph_range, consensus_check, combined_formation, combined_tracked_edges, desired_wb_formation, desired_slot_formation) -> None:
        ### Simulation plot set up
        self.graph_range = graph_range
        self.square_range = graph_range/2 # Utilized when completing a square trajectory so that all agents are still visible on the map
        self.ax = ax
        self.ax.axis([-self.graph_range,self.graph_range,-self.graph_range,self.graph_range])

        ### Set up vars
        self.dt = dt
        self.desired_wb_formation = desired_wb_formation
        self.desired_slot_formation = desired_slot_formation
        self.tracked_edges = tracked_edges

        ### Initialize simulation plot points
        self.wb_points = []
        for i in range(len(wb_init_locs)):
            point, = self.ax.plot(wb_init_locs[i][0],wb_init_locs[i][1], marker="o")
            self.wb_points.append(point)
        
        self.slot_points = []
        for i in range(len(slot_init_locs)):
            point, = self.ax.plot(slot_init_locs[i][0],slot_init_locs[i][1], marker="o")
            self.slot_points.append(point)

        ### Combined formation vars
        self.consensus_check = consensus_check
        self.combined_formation = combined_formation
        self.combined_tracked_edges = combined_tracked_edges
        self.combined_edge_lengths = []

    def add_wb_agent(self, agent, init_loc) -> None:
        self.wb_agents.append(agent)
        self.wb_agent_locs.append(init_loc)

    def add_slot_agent(self, agent, init_loc) -> None:
        self.slot_agents.append(agent)
        self.slot_agent_locs.append(init_loc)

    def step(self,_): # Returns updated plot points       
        #################################################################### 
        # Send msg updates
        ####################################################################

        ###If consensus has not been reached between slot and wb formations
        if not self.consensus_check:
            ### Weight Based message updates
            for cur_agent in range(len(self.wb_agents)):
                ### Determine what neighbors of the current agent are within the comms_radius
                current_loc = self.wb_agent_locs[cur_agent]
                diffs = np.array(self.wb_agent_locs) - np.array(current_loc)
                mags = np.linalg.norm(diffs, axis=1)
                in_comms_range_indices = np.argwhere(mags < self.wb_agents[cur_agent].comms_radius)

                for neighbor_agent in list(in_comms_range_indices):
                    ### Only append messages to neighbor message lists
                    if cur_agent != neighbor_agent[0] and neighbor_agent[0] in self.tracked_edges[cur_agent]: 
                        msg = self.wb_agents[cur_agent].create_msg(self.wb_agent_locs[cur_agent],cur_agent)
                        self.wb_agents[neighbor_agent[0]].append_msg(msg)    

                ### Plot errors
                # Postional error
                self.wb_agents[cur_agent].set_pos_error(0)

                # Distance error
                if cur_agent == 0:
                    self.wb_agents[cur_agent].set_dist_error(0) # Leader does not have error, only responsible for moving towards future point
                else:
                    actual_dist = ((self.wb_agent_locs[cur_agent-1][0] - self.wb_agent_locs[cur_agent][0])**2 + (self.wb_agent_locs[cur_agent-1][1] - self.wb_agent_locs[cur_agent][1])**2)**(1/2)
                    dist_error = self.wb_agents[cur_agent].get_edge_length()[cur_agent - 1] - actual_dist
                    self.wb_agents[cur_agent].set_dist_error(dist_error)

            ### Slot Based message updates
            for cur_agent in range(len(self.slot_agents)):
                ### Set current location
                self.slot_agents[cur_agent].set_cur_loc(self.slot_agent_locs[cur_agent],cur_agent)

                ### Plot errors
                # Distance error
                if self.slot_agents[cur_agent].is_leader:
                    self.slot_agents[cur_agent].set_dist_error(0) # Leader does not have error, only responsible for moving towards future point
                else:
                    actual_dist = ((self.slot_agent_locs[cur_agent-1][0] - self.slot_agent_locs[cur_agent][0])**2 + (self.slot_agent_locs[cur_agent-1][1] - self.slot_agent_locs[cur_agent][1])**2)**(1/2)
                    dist_error = self.slot_agents[cur_agent].get_edge_length()[cur_agent - 1] - actual_dist
                    self.slot_agents[cur_agent].set_dist_error(dist_error)

                if self.slot_agents[cur_agent].is_leader:
                    ### Calculate slot agents within comms radius
                    current_loc = self.slot_agent_locs[cur_agent]
                    diffs = np.array(self.slot_agent_locs) - np.array(current_loc)
                    mags = np.linalg.norm(diffs, axis=1)
                    in_slot_comms_range_indices = np.argwhere(mags < self.slot_agents[cur_agent].comms_radius)

                    for neighbor_agent in list(in_slot_comms_range_indices):
                        if cur_agent != neighbor_agent[0]: # Only send messages from the leader to follower agents for the slot offset formation
                            leader_heading = self.slot_agents[cur_agent].calc_leader_heading()
                            msg = self.slot_agents[cur_agent].create_leader_msg(self.slot_agent_locs[cur_agent],cur_agent,leader_heading)
                            self.slot_agents[neighbor_agent[0]].append_msg(msg)  

                    ####################################################################
                    # Verify if formations have reached consensus.
                    # The slot leader verifies if it can communicate with every wb agent.
                    # The slot leader uses its comms radius to check what wb agents are in range.
                    ####################################################################

                    ### Calculate weight based agents within comms radius
                    diffs = np.array(self.wb_agent_locs) - np.array(current_loc)
                    mags = np.linalg.norm(diffs, axis=1)
                    in_wb_comms_range_indices = np.argwhere(mags < self.slot_agents[cur_agent].comms_radius)

                    if (len(in_slot_comms_range_indices) + len(in_wb_comms_range_indices)) == (len(self.wb_agents) + len(self.slot_agents)):
                        self.consensus_check = True

                        ### Leader of the weight based needs to be set to False
                        self.wb_agents[0].set_leader(False) # Assuming agent 0 is the leader

                        ### Trajectory for the slot offset formation leader, wb agents will follow this trajectory
                        # self.slot_agents[0].circle_trajectory(200, 100) # Assuming agent 0 is the leader
                        # self.slot_agents[0].line_trajectory()
                        self.slot_agents[0].square_trajectory(self.square_range)

                        ### Update slot locations for combined formation
                        for cur_agent in range(len(self.slot_agents)):
                            scaled_slot_pos = np.array(self.combined_formation[cur_agent]) * combined_scale_value
                            self.slot_agents[cur_agent].set_slot_loc(scaled_slot_pos)

                        ### Update tracked edges for the combined formation
                        edge_calcs = np.zeros([len(self.combined_formation),len(self.combined_formation)])

                        for cur_agent in range(len(self.combined_formation)):
                            for neighbor_agent in range(len(self.combined_formation)):
                                edge_calcs[cur_agent][neighbor_agent] = (((self.combined_formation[cur_agent][0] - self.combined_formation[neighbor_agent][0])**2)+((self.combined_formation[cur_agent][1] - self.combined_formation[neighbor_agent][1])**2))**(1/2)
                        
                        total_agents = len(self.slot_agents) + len(self.wb_agents)
                        for cur_agent in range(total_agents): 
                            edge_lengths = []
                            
                            for neighbor_agent in range(len(self.combined_formation)):
                                edge_lengths.append(np.array([edge_calcs[cur_agent][neighbor_agent]]) * combined_scale_value)
                            
                            edge_lengths = list(edge_lengths)
                            self.combined_edge_lengths.append(edge_lengths)

                            if cur_agent >= len(self.slot_agents):
                                self.wb_agents[(cur_agent - len(self.slot_agents))].set_edge_length(edge_lengths)
                        
        else: # Once the slot and wb formations have combined into one formation          
            ### Messages being sent from offset and weight based need to be combined
            # Weight Based message updates
            for cur_agent in range(len(self.wb_agents)):
                ### Determine what neighbors of the current agent are within the comms_radius
                # Wb agents in range
                current_loc = self.wb_agent_locs[cur_agent]
                diffs = np.array(self.wb_agent_locs) - np.array(current_loc)
                mags = np.linalg.norm(diffs, axis=1)
                in_wb_comms_range_indices = np.argwhere(mags < self.wb_agents[cur_agent].comms_radius)

                # Slot agents in range
                diffs = np.array(self.slot_agent_locs) - np.array(current_loc)
                mags = np.linalg.norm(diffs, axis=1)
                in_slot_comms_range_indices = np.argwhere(mags < self.wb_agents[cur_agent].comms_radius)

                ### Create and append messages from neighbor agents within comms range to current agents message queue
                for neighbor_agent in list(in_wb_comms_range_indices):
                    if cur_agent != neighbor_agent and (neighbor_agent + len(self.slot_agents)) in self.combined_tracked_edges[cur_agent]: 
                        msg = self.wb_agents[neighbor_agent[0]].create_msg(self.wb_agent_locs[neighbor_agent[0]],neighbor_agent[0]+len(self.slot_agents))
                        self.wb_agents[cur_agent].append_msg(msg) 
                        
                for neighbor_agent in list(in_slot_comms_range_indices):
                    if neighbor_agent in self.combined_tracked_edges[cur_agent]:
                        msg = self.slot_agents[neighbor_agent[0]].create_msg(self.slot_agent_locs[neighbor_agent[0]],neighbor_agent[0])
                        self.wb_agents[cur_agent].append_msg(msg) 

            # Slot Based message updates
            leader_heading = 0 # place holder for positional error calculations
            for cur_agent in range(len(self.slot_agents)):
                ### Set current location
                self.slot_agents[cur_agent].set_cur_loc(self.slot_agent_locs[cur_agent],cur_agent)

                if self.slot_agents[cur_agent].is_leader:
                    ### Calculate slot agents within comms radius
                    current_loc = self.slot_agent_locs[cur_agent]
                    diffs = np.array(self.slot_agent_locs) - np.array(current_loc)
                    mags = np.linalg.norm(diffs, axis=1)
                    in_slot_comms_range_indices = np.argwhere(mags < self.slot_agents[cur_agent].comms_radius)

                    for neighbor_agent in list(in_slot_comms_range_indices):
                        if cur_agent != neighbor_agent:
                            leader_heading = self.slot_agents[cur_agent].calc_leader_heading()
                            self.slot_agents[neighbor_agent[0]].append_msg(self.slot_agents[cur_agent].create_leader_msg(self.slot_agent_locs[cur_agent],cur_agent,leader_heading))  
            
            ### Plot errror calculations

            # Positional error
            for cur_agent in range(len(self.wb_agent_locs)):
                ### Get the error in position based on leader position and formation slot
                leader_heading_rads = math.radians(leader_heading - 90) # Subtract 90 so that slots shift as if leader is at 0,0
                goal_x = self.slot_agent_locs[0][0] - (math.cos(leader_heading_rads) * self.combined_formation[cur_agent+len(self.slot_agents)][0]*wb_scale_value - math.sin(leader_heading_rads) * self.combined_formation[cur_agent+len(self.slot_agents)][1] * wb_scale_value)
                goal_y = self.slot_agent_locs[0][1] - (math.sin(leader_heading_rads) * self.combined_formation[cur_agent+len(self.slot_agents)][0]*wb_scale_value + math.cos(leader_heading_rads) * self.combined_formation[cur_agent+len(self.slot_agents)][1] * wb_scale_value)
                goal_loc = np.array((goal_x,goal_y))
                
                diff = goal_loc - self.wb_agent_locs[cur_agent]
                pos_error = (diff[0]**2 + diff[1]**2)**(1/2)
                self.wb_agents[cur_agent].set_pos_error(pos_error)
            
            # Distance error
            for cur_agent in range(len(combined_formation)):
                if cur_agent == 0:
                    self.slot_agents[cur_agent].set_dist_error(0)
                else:
                    if cur_agent < len(self.slot_agents): # Prev is a slot and cur is a slot agent
                        actual_dist = ((self.slot_agent_locs[cur_agent-1][0] - self.slot_agent_locs[cur_agent][0])**2 + (self.slot_agent_locs[cur_agent-1][1] - self.slot_agent_locs[cur_agent][1])**2)**(1/2)
                        dist_error = self.combined_edge_lengths[cur_agent][cur_agent-1][0] - actual_dist
                        print(f"Cur agent {cur_agent} dist error is: {dist_error}")
                        print(f"actual: {actual_dist}. combined edge dist: {self.combined_edge_lengths[cur_agent][cur_agent-1][0]}")
                        self.slot_agents[cur_agent].set_dist_error(dist_error)
                    elif cur_agent == len(self.slot_agents): # Prev is a slot agent and cur is a wb agent
                        actual_dist = ((self.slot_agent_locs[cur_agent-1][0] - self.wb_agent_locs[cur_agent-len(self.slot_agents)][0])**2 + (self.slot_agent_locs[cur_agent-1][1] - self.wb_agent_locs[cur_agent-len(self.slot_agents)][1])**2)**(1/2)
                        dist_error = self.combined_edge_lengths[cur_agent][cur_agent-1][0]-actual_dist
                        self.wb_agents[cur_agent - len(self.slot_agents)].set_dist_error(dist_error)
                    else: # Prev is a wb agent an cur is a wb agent
                        print(f"Cur agent is: {cur_agent}. Prev agent is: {cur_agent-1 - len(self.slot_agents)}")
                        print(f"WB agent locs: {self.wb_agent_locs}")
                        actual_dist = ((self.wb_agent_locs[cur_agent-1 - len(self.slot_agents)][0] - self.wb_agent_locs[cur_agent-len(self.slot_agents)][0])**2 + (self.wb_agent_locs[cur_agent-1 - len(self.slot_agents)][1] - self.wb_agent_locs[cur_agent-len(self.slot_agents)][1])**2)**(1/2)
                        dist_error = self.combined_edge_lengths[cur_agent][cur_agent-1][0]-actual_dist
                        self.wb_agents[cur_agent - len(self.slot_agents)].set_dist_error(dist_error)


        ####################################################################
        # Store the current velocity for the given agents
        ####################################################################

        ### Weight based velocities
        wb_agent_vels = []
        for i in range(len(self.wb_agents)):
            wb_agent_vels.append(self.wb_agents[i].process_msgs())

        ### Slot velocities
        slot_agent_vels = []
        for i in range(len(self.slot_agents)):
            slot_agent_vels.append(self.slot_agents[i].process_msgs())

        ####################################################################
        # Update agent locations based on the calculated velocities
        ####################################################################

        ### Weight based location updates
        for cur_agent in range(len(self.wb_agents)):
            cur_x_pos = self.wb_agent_locs[cur_agent][0]
            cur_y_pos = self.wb_agent_locs[cur_agent][1]

            self.wb_agent_locs[cur_agent][0] = cur_x_pos + wb_agent_vels[cur_agent][0] * self.dt
            self.wb_agent_locs[cur_agent][1] = cur_y_pos + wb_agent_vels[cur_agent][1] * self.dt

        ### Slot location updates
        for cur_agent in range(len(self.slot_agents)):
            cur_x_pos = self.slot_agent_locs[cur_agent][0]
            cur_y_pos = self.slot_agent_locs[cur_agent][1]

            self.slot_agent_locs[cur_agent][0] = cur_x_pos + slot_agent_vels[cur_agent][0] * self.dt
            self.slot_agent_locs[cur_agent][1] = cur_y_pos + slot_agent_vels[cur_agent][1] * self.dt

        #######################################################################
        # Update the points and annotations for the simulation plot
        #######################################################################
        point_annotations = []

        ### Weight based point updates
        for cur_agent in range(len(self.wb_agents)):
            self.wb_points[cur_agent].set_data([self.wb_agent_locs[cur_agent][0]],[self.wb_agent_locs[cur_agent][1]])
            point_annotations.append(self.ax.text(self.wb_agent_locs[cur_agent][0],self.wb_agent_locs[cur_agent][1], f'WB{cur_agent}', fontsize=10, ha='right'))
        
        ### Slot based point updates
        for cur_agent in range(len(self.slot_agents)):
            self.slot_points[cur_agent].set_data([self.slot_agent_locs[cur_agent][0]],[self.slot_agent_locs[cur_agent][1]])
            point_annotations.append(self.ax.text(self.slot_agent_locs[cur_agent][0],self.slot_agent_locs[cur_agent][1], f'SL{cur_agent}', fontsize=10, ha='right'))

        ####################################################################
        # Return the points and annotations
        ####################################################################
        return self.wb_points + self.slot_points + point_annotations

    def plot_error(self):
        ### Retrieve data
        num_slot_agents = len(self.slot_agents)
        num_wb_agents = len(self.wb_agents)
        slot_pos_errors = []
        slot_dist_errors = []
        wb_pos_errors = []
        wb_dist_errors = []
        
        for cur_agent in range(num_slot_agents):
            slot_pos_errors.append(self.slot_agents[cur_agent].get_pos_error())
            slot_dist_errors.append(self.slot_agents[cur_agent].get_dist_error())
        
        for cur_agent in range(num_wb_agents):
            wb_pos_errors.append(self.wb_agents[cur_agent].get_pos_error())
            wb_dist_errors.append(self.wb_agents[cur_agent].get_dist_error())

        num_points = len(slot_pos_errors[0])
        time = np.linspace(0, num_points * dt, num_points)

        ### Plotting
        plt.figure(figsize=(10, 6))
        
        for i in range(num_slot_agents):
            plt.plot(time, slot_dist_errors[i], label=f'Slot Offset Agent {i}')
        
        for i in range(num_wb_agents):
            plt.plot(time, wb_dist_errors[i], label=f'Weight-Based Agent {i}')

        ### 
        # Adding annotations for points where error values are greater than threshold
        # Going to leave commented out for now... doesn't look great, will add annotations manually
        ###
        # threshold = 100
        # plt.annotate(f'Initial agent movements of \n random seed location to \n formation position', xy=(time[0], slot_dist_errors[-1][0]), xytext=(time[0], slot_dist_errors[-1][0] - 20),
        #             arrowprops=dict(facecolor='red', arrowstyle='->'))
        # within_threshold = True
        # for i in range(num_slot_agents):
        #     for j in range(num_points):
        #         if abs(wb_dist_errors[i][j]) > threshold:
        #             if j < 10:
        #                 continue
        #             if within_threshold is True:
        #                 plt.annotate(f'Change in formation positions due to direction or combination', xy=(time[j], slot_dist_errors[i][j]), xytext=(time[j], slot_dist_errors[i][j] - 10),
        #                             arrowprops=dict(facecolor='red', arrowstyle='->'))
        #                 within_threshold = False
        #         else:
        #             within_threshold = True
        
        # within_threshold = True
        # for i in range(num_wb_agents):
        #     for j in range(num_points):
        #         if abs(wb_dist_errors[i][j]) > threshold:
        #             if j < 10:
        #                 continue
        #             if within_threshold is True:
        #                 plt.annotate(f'Change in formation positions due to direction or combination', xy=(time[j], wb_dist_errors[i][j]), xytext=(time[j], wb_dist_errors[i][j] - 10),
        #                             arrowprops=dict(facecolor='red', arrowstyle='->'))
        #                 within_threshold = False
        #         else:
        #             within_threshold = True

        ### Labels
        plt.xlabel('Comms Passes (Number of Steps * dt)')
        plt.ylabel('Interagent Distances Based on Topology')
        plt.title('Interagent Distances Based on Topology Over Comms Passes')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":

    ####################################################################
    # Things you must specify for the simulation
    ####################################################################

    ####################################################################
    # Plot and step Initialization
    ####################################################################
    ### Step and plot specific
    dt = 0.005
    graph_range = 500

    ### Random seed values
    rendezvous_seed = 75
    wb_locs_seed = 555
    slot_locs_seed = 38

    ### Rendezvous point for combined formations
    random.seed(rendezvous_seed)
    min_point = random.randint(-graph_range + 100, graph_range - 100)
    rendezvous_x = random.uniform(-min_point,min_point)
    rendezvous_y = random.uniform(-min_point,min_point)
    rendezvous_pos = [rendezvous_x, rendezvous_y]

    ### Combined formation initial slot positions and wb edge tracking
    
    # Bowling bin combined formation
    num_wb_agents = 3
    num_slot_agents = 3
    combined_scale_value = 80
    combined_formation = [[0,0],[-1,-1],[1,-1],[-2,-2],[0,-2],[2,-2]]
    combined_tracked_edges = {0:[1,2,4,5], 1:[0,1,2,3,5], 2:[1,2,3,4]}

    # V shaped combined formation
    # num_wb_agents = 4
    # num_slot_agents = 3
    # combined_scale_value = 50
    # combined_formation = [[0,0],[1,-1],[-1,-1],[-2,-2],[2,-2],[-3,-3],[3,-3]]
    # combined_tracked_edges = {0:[0,1,2], 1:[0,1,2], 2:[0,1,2], 3:[0,1,2]} # This gets the right shape, but agents are not in the correct order

    # Square combined formation
    num_wb_agents = 5
    num_slot_agents = 3
    combined_scale_value = 100
    combined_formation = [[0,0],[-1,0],[1,0],[-1,-1],[-1,-2],[0,-2],[1,-1],[1,-2]]
    combined_tracked_edges = {0:[1,4,6,2,7], 1:[1,7,2,3,5], 2:[4,7,1,0,2], 3:[2,7,3,1,4], 4:[2,4,1,5,6]}

    # Star combined formation
    # num_wb_agents = 6
    # num_slot_agents = 3
    # combined_scale_value = 20
    # combined_formation = [[0,0],[2,-5],[-2,-5],[7,-5],[-7,-5],[3,-8],[-3,-8],[0,-10],[4.5,-13.5]]
    # combined_tracked_edges = {0:[1,4,6,2,7], 1:[1,7,2,3,5], 2:[4,7,1,0,2], 3:[2,7,3,1,4], 4:[2,4,1,5,6], 5:[2,6,3,1,4], 6:[2,4,1,5,3]}


    ####################################################################
    # Weight Based Initialization
    ####################################################################

    ### Agent specific - locations and num agents
    comms_wb_radius = 10000 
    start_radius = 200
    
    init_wb_locs = []
    random.seed(wb_locs_seed)
    min_point = random.randint(-graph_range + 100, graph_range - 100) # minimum range for init locs
    for _ in range(num_wb_agents):
        x = random.uniform(min_point - (start_radius/2), min_point + (start_radius/2)) # min_point +/- 0.5 * comms radius
        y = random.uniform(min_point - (start_radius/2), min_point + (start_radius/2))
        init_wb_locs.append([x, y])

    ### Follower Speed 
    max_wb_follower_speed = 40

    ### Leader Specific
    wb_leader_idx = 0 # Start indexing at 0
    max_wb_leader_speed = 15
    wb_trajectory_type = "circle"

    ### Desired formation and associated agent locations that can be scaled
    bp_formation = np.array([[0,0],[-1,-1],[1,-1],[-2,-2],[2,-2],[-3,-3],[-1,-3],[1,-3],[3,-3]])
    desired_wb_formation = bp_formation
    wb_scale_value = 75
    scaled_wb_formation = desired_wb_formation * wb_scale_value

    ### Associated edges between agents, ex. agent 0 has connections with agents 1, 2, and 4 if 0:[1,2,4]
    tracked_edges = {0:[1,2,4], 1:[0,2,3,4,5], 2:[0,1,3,4,5], 3:[1,2,4,5], 4:[0,1,2,3,5], 5:[1,2,3,4], 6:[0,1,2,3,4]}

    ####################################################################
    # Slot Initialization
    ####################################################################

    ### Agent specific
    comms_slot_radius = 500 
    start_radius = 200

    init_slot_locs = []
    random.seed(slot_locs_seed)
    min_point = random.randint(-graph_range + 100, graph_range - 100) # minimum range for init locs
    for _ in range(num_slot_agents):
        x = random.uniform(min_point - (start_radius/2), min_point + (start_radius/2)) # min_point +/- 0.5 * comms radius
        y = random.uniform(min_point - (start_radius/2), min_point + (start_radius/2))
        init_slot_locs.append([x, y])

    ### Follower Speed
    max_slot_follower_speed = 30
    follower_k = 1
    
    ### Leader Specific
    slot_leader_idx = 0 # Start indexing at 0
    max_slot_leader_speed = 10
    slot_trajectory_type = "circle"

    ### Desired formation and associated agent locations that can be scaled
    bp_formation = np.array([[0,0],[-1,-1],[1,-1]])
    # v_formation = np.array([[0,0],[-1,-1],[1,-1],[-2,-2],[2,-2]])
    desired_slot_formation = bp_formation
    slot_scale_value = 75
    scaled_slot_formation = desired_slot_formation * slot_scale_value

    ####################################################################
    ####################################################################
    
    ### Plot set up
    fig,ax = plt.subplots()
    ax.set_title("Weight-Based and Offset Formation Control")

    ### Initialize the container for the simulation
    consensus_check = False
    sim_container = Container(init_wb_locs, init_slot_locs, dt, ax, tracked_edges, graph_range, consensus_check, combined_formation, combined_tracked_edges, scaled_wb_formation, scaled_slot_formation)

    ### Add WEIGHT-BASED agents to the simulation and calculate edges
    edge_calcs = np.zeros([len(desired_wb_formation),len(desired_wb_formation)])
    for cur_agent in range(len(desired_wb_formation)):
        for neighbor_agent in range(len(desired_wb_formation)):
            edge_calcs[cur_agent][neighbor_agent] = (((desired_wb_formation[cur_agent][0] - desired_wb_formation[neighbor_agent][0])**2)+((desired_wb_formation[cur_agent][1] - desired_wb_formation[neighbor_agent][1])**2))**(1/2)

    for cur_agent in range(num_wb_agents):
        edge_lengths = []
        for neighbor_agent in range(len(desired_wb_formation)):
            edge_lengths.append(edge_calcs[cur_agent][neighbor_agent] * wb_scale_value)
        
        edge_lengths = list(edge_lengths)

        if cur_agent == wb_leader_idx:
            cur_wb_agent = WB_Agent(comms_wb_radius,edge_lengths,max_wb_follower_speed,max_wb_leader_speed,True)
            cur_wb_agent.init_goal(init_wb_locs[cur_agent], rendezvous_pos, wb_trajectory_type, 5) # Make it so after they get to their point, they circle
            sim_container.add_wb_agent(cur_wb_agent,init_wb_locs[cur_agent])
        else:
            sim_container.add_wb_agent(WB_Agent(comms_wb_radius,edge_lengths,max_wb_follower_speed,max_wb_leader_speed,False),init_wb_locs[cur_agent])

    ### Add SLOT-BASED agents to the simulation and calculate edges for plot
    edge_calcs = np.zeros([len(desired_slot_formation),len(desired_slot_formation)])
    for cur_agent in range(len(desired_slot_formation)):
        for neighbor_agent in range(len(desired_slot_formation)):
            edge_calcs[cur_agent][neighbor_agent] = (((desired_slot_formation[cur_agent][0] - desired_slot_formation[neighbor_agent][0])**2)+((desired_slot_formation[cur_agent][1] - desired_slot_formation[neighbor_agent][1])**2))**(1/2)

    for cur_agent in range(num_slot_agents):
        edge_lengths = []
        for neighbor_agent in range(len(desired_slot_formation)):
            edge_lengths.append(edge_calcs[cur_agent][neighbor_agent] * slot_scale_value)
        
        edge_lengths = list(edge_lengths)

        if cur_agent == slot_leader_idx:
            cur_slot_agent = Slot_Agent(comms_slot_radius,edge_lengths,scaled_slot_formation[cur_agent],follower_k,max_slot_leader_speed,True)
            cur_slot_agent.init_goal(init_slot_locs[cur_agent], rendezvous_pos, slot_trajectory_type, 5)
            sim_container.add_slot_agent(cur_slot_agent,init_slot_locs[cur_agent])
        else:
            sim_container.add_slot_agent(Slot_Agent(comms_slot_radius,edge_lengths,scaled_slot_formation[cur_agent],follower_k,max_slot_follower_speed,False),init_slot_locs[cur_agent])

    ### Animate the simulation
    ani = FuncAnimation(fig, sim_container.step, interval=dt, blit=True, repeat=True, cache_frame_data=False)

    plt.show()

    ### Plot the error curves
    sim_container.plot_error()

        