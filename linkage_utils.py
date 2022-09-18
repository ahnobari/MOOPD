from __future__ import division
from math import sin, cos, acos, pi
import numpy as np
import pymoo
from tqdm.autonotebook import trange
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pymoo.model.problem import FunctionalProblem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.model.callback import Callback
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
import multiprocessing as mp
from tqdm import tqdm
from scipy.stats.kde import gaussian_kde
import PIL
import warnings
from scipy.ndimage.filters import gaussian_filter
warnings.filterwarnings("ignore")
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
import pickle
import json
import xml.etree.ElementTree as etree

from svgpath2mpl import parse_path


def chamfer_distance(x, y, metric='l2', direction='bi', subsample=True, n_max=250):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    if subsample:
        if x.shape[0]>n_max:
            x_s = x
        else:
            x_s = x[np.round(np.linspace(0,x.shape[0]-1,n_max)).astype(np.int32)]
        if y.shape[0]>n_max:
            y_s = y
        else:
            y_s = y[np.round(np.linspace(0,y.shape[0]-1,n_max)).astype(np.int32)]
        return chamfer_distance(x_s,y_s,subsample=False)
            
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

def line_segment(start, end):
    """Bresenham's Line Algorithm
    """
    # Setup initial conditions
    x1, y1 = start[0], start[1]
    x2, y2 = end[0], end[1]
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = np.abs(dy) > np.abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

def rasterized_curve_coords(curve, res):
    c = np.minimum(res-1,np.floor(curve*res)).astype(np.int32)
    ps = []
    for i in range(curve.shape[0]-1):
        ps += line_segment(c[i],c[i+1])
    
#     out = []
    
#     for p in ps:
#         if not p in out:
#             out.append(p)
    
#     out = np.array(out)
    out = np.array(list(set(ps)))
#     i = np.where(out[:,0] == res-1)[0][0] + np.argmin(out[np.where(out[:,0] == res-1)],0)[1]
#     out = np.concatenate([out[i:],out[:i]])
    
    return out

def rasterize_curve(curve, res):
    c = np.minimum(res-1,np.floor(curve*res)).astype(np.int32)
    ps = []
    for i in range(curve.shape[0]-1):
        ps += line_segment(c[i],c[i+1])
    
    ps = np.array(list(set(ps)))
    out = np.zeros([res,res]).astype(np.bool)
    out[ps[:,1],ps[:,0]] = True
    
    return np.flipud(out)

# def draw_mechanism(C,x,fixed_nodes,motor):
#     fig = plt.figure(figsize=(12,12))
#     N = C.shape[0]
#     for i in range(N):
#         if i in fixed_nodes:
#             plt.scatter(x[i,0],x[i,1],color="Black",s=100,zorder=10)
#         else:
#             plt.scatter(x[i,0],x[i,1],color="Grey",s=100,zorder=10)
        
#         for j in range(i+1,N):
#             if C[i,j]:
#                 plt.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="Black",linewidth=3.5)
#     solver = solver_cpu()
#     xo = solver.solve_rev(192,x,C,motor,fixed_nodes,False)
#     if not np.array(xo).size == 1:
#         for i in range(C.shape[0]):
#             if not i in fixed_nodes:
#                 plt.plot(xo[:,i,0],xo[:,i,1])
#     else:
#         plt.text(0.5, 0.5, 'Locking Or Under Defined', color='red', horizontalalignment='center', verticalalignment='center')
        
#     plt.axis('equal')
#     plt.xlim([0,1])
#     plt.ylim([0,1])
     
                
def animate_mechanims(C,xs,fixed_nodes):
    fig = plt.figure()
    draw_mechanism(C,xs[0],fixed_nodes)
    plt.xlim([-0.2,1.2])
    plt.ylim([-0.2,1.2])
    def update_mechanism_fig(index):
        plt.cla()
        draw_mechanism(C,xs[index],fixed_nodes)
        plt.xlim([-0.2,1.2])
        plt.ylim([-0.2,1.2])
    anim = animation.FuncAnimation(fig, update_mechanism_fig, 192,interval=20)
    
class solver_cpu():
    def __init__(self):
        pass
    
    def find_neighbors(self, index, C):
        return np.where(C[index])[0]
    
    def find_unvisited_neighbors(self, index, C, visited_list):
        return np.where(np.logical_and(C[index],np.logical_not(visited_list)))[0]
    
    def get_G(self, x0, C):
        N = C.shape[0]
        G = np.zeros_like(C, dtype=float)
        for i in range(N):
            for j in range(i):
                if C[i,j]:
                    G[i,j] = G[j,i] = np.linalg.norm(x0[i]-x0[j])
                else:
                    G[i,j] = G[j,i] = np.inf
                    
        return G
    
    def get_path(self, x0, C, G, motor, fixed_nodes=[0, 1], show_msg=False):
        
        theta = 0.0
        
        path = []
        op = []
        
        N = C.shape[0]
        x = np.zeros((N, 2))
        visited_list = np.zeros(N, dtype=bool)
        active_list = []
        
        visited_list[fixed_nodes] = True
        visited_list[motor[1]] = True
        
        x[fixed_nodes] = x0[fixed_nodes]
        
        dx = x0[motor[1],0] - x0[motor[0],0]
        dy = x0[motor[1],1] - x0[motor[0],1]
        
        theta_0 = np.math.atan2(dy,dx)
        theta = theta_0 + theta
        
        x[motor[1], 0] = x[motor[0],0] + G[motor[0],motor[1]] * np.cos(theta)
        x[motor[1], 1] = x[motor[0],1] + G[motor[0],motor[1]] * np.sin(theta)
        
        for i in np.where(visited_list)[0]:            
            active_list += list(self.find_unvisited_neighbors(i, C, visited_list))
            
        active_list = list(set(active_list))
        
        counter = 0
        
        while len(active_list)>0:
            k = active_list.pop(0)
            neighbors = self.find_neighbors(k, C)
            vn = neighbors[visited_list[neighbors]]
            if vn.shape[0]>1:
                if vn.shape[0]>2 and show_msg:
                    print('Redudndant or overdefined system.')
                i = vn[0]
                j = vn[1]
                l_ij = np.linalg.norm(x[j]-x[i])
                s = np.sign((x0[i,1]-x0[k,1])*(x0[i,0]-x0[j,0]) - (x0[i,1]-x0[j,1])*(x0[i,0]-x0[k,0]))
                cosphi = (l_ij**2+G[i,k]**2-G[j,k]**2)/(2*l_ij*G[i,k])
                if cosphi >= -1.0 and cosphi <= 1.0:
                    phi = s * acos(cosphi)
                    R = np.array([[cos(phi), -sin(phi)],
                                  [sin(phi), cos(phi)]])
                    scaled_ij = (x[j]-x[i])/l_ij * G[i,k]
                    x[k] = np.matmul(R, scaled_ij.reshape(2,1)).flatten() + x[i]
                    path.append(k)
                    op.append([i,j,s])
                else:
                    if show_msg:
                        print('Locking or degenerate linkage!')
                    return None
                
                visited_list[k] = True
                active_list += list(self.find_unvisited_neighbors(k, C, visited_list))
                active_list = list(set(active_list))
                counter = 0
            else:
                counter += 1
                active_list.append(k)
            
            if counter > len(active_list):
                if show_msg:
                    print('DOF larger than 1.')
                return None
        return path, op
    
    def position_from_path(self, path, op, theta, x0, C, G, motor, fixed_nodes=[0, 1], show_msg=False):
        
        N = C.shape[0]
        x = np.zeros((N, 2))
        visited_list = np.zeros(N, dtype=bool)
        active_list = []
        
        visited_list[fixed_nodes] = True
        visited_list[motor[1]] = True
        
        x[fixed_nodes] = x0[fixed_nodes]
        
        dx = x0[motor[1],0] - x0[motor[0],0]
        dy = x0[motor[1],1] - x0[motor[0],1]
        
        theta_0 = np.math.atan2(dy,dx)
        theta = theta_0 + theta
        
        x[motor[1], 0] = x[motor[0],0] + G[motor[0],motor[1]] * np.cos(theta)
        x[motor[1], 1] = x[motor[0],1] + G[motor[0],motor[1]] * np.sin(theta)
        
        
        for l,step in enumerate(path):
            i = op[l][0]
            j = op[l][1]
            k = step
            
            l_ij = np.linalg.norm(x[j]-x[i])
            cosphi = (l_ij**2+G[i,k]**2-G[j,k]**2)/(2*l_ij*G[i,k])
            if cosphi >= -1.0 and cosphi <= 1.0:
                phi = op[l][2] * acos(cosphi)
                R = np.array([[cos(phi), -sin(phi)],
                              [sin(phi), cos(phi)]])
                scaled_ij = (x[j]-x[i])/l_ij * G[i,k]
                x[k] = np.matmul(R, scaled_ij.reshape(2,1)).flatten() + x[i]
            else:
                if show_msg:
                    print('Locking or degenerate linkage!')
                return np.abs(cosphi)
        return x
    
    def check(self, n_steps, x0, C, motor, fixed_nodes=[0, 1], show_msg=False, lim=[0.0,1.0]):
        G = self.get_G(x0,C)
        
        pop = self.get_path(x0, C, G, motor, fixed_nodes, show_msg)
        
        if not pop:
            return False
        
        func = lambda t: self.position_from_path(pop[0],pop[1],t, x0, C, G, motor, fixed_nodes, show_msg)
        
        ts = np.linspace(0, 2*np.pi, n_steps)
        
        out = []
        
        for t in ts:
            x_temp = func(t)
            if np.array(x_temp).size == 1:
                return False
            else:
                out.append(x_temp)
            
        if np.max(out) <= lim[1] and np.min(out) >= lim[0]:
            return True
        else:
            return False
    
    def solve_rev(self, n_steps, x0, C, motor, fixed_nodes=[0, 1], show_msg=False):
        
        G = self.get_G(x0,C)
        
        pop = self.get_path(x0, C, G, motor, fixed_nodes, show_msg)

        if not pop:
            return 10
        
        func = lambda t: self.position_from_path(pop[0],pop[1],t, x0, C, G, motor, fixed_nodes, show_msg)
        
        ts = np.linspace(0, 2*np.pi, n_steps)
        
        out = []
        
        for t in ts:
            x_temp = func(t)
            if np.array(x_temp).size == 1:
                if x_temp:
                    return np.abs(x_temp)
                else:
                    return None
            else:
                out.append(x_temp)
        
        return np.array(out)
    
    def material(self,x0, C):
        G = self.get_G(x0,C)
        return np.sum(G[np.logical_not(np.isinf(G))])

class mechanism_solver():
    def __init__(self):
        pass
    
    def find_neighbors(self, index, C):
        return np.where(C[index])[0]
    
    def find_unvisited_neighbors(self, index, C, visited_list):
        return np.where(np.logical_and(C[index],np.logical_not(visited_list)))[0]
    
    def get_G(self, x0, C):
        N = C.shape[0]
        G = np.zeros_like(C, dtype=float)
        for i in range(N):
            for j in range(i):
                if C[i,j]:
                    G[i,j] = G[j,i] = np.linalg.norm(x0[i]-x0[j])
                else:
                    G[i,j] = G[j,i] = np.inf
                    
        return G
    
    def get_path(self, x0, C, G, motor, fixed_nodes=[0, 1], show_msg=False):
        
        theta = 0.0
        
        path = []
        op = []
        
        N = C.shape[0]
        x = np.zeros((N, 2))
        visited_list = np.zeros(N, dtype=bool)
        active_list = []
        
        visited_list[fixed_nodes] = True
        visited_list[motor[1]] = True
        
        x[fixed_nodes] = x0[fixed_nodes]
        
        dx = x0[motor[1],0] - x0[motor[0],0]
        dy = x0[motor[1],1] - x0[motor[0],1]
        
        theta_0 = np.math.atan2(dy,dx)
        theta = theta_0 + theta
        
        x[motor[1], 0] = x[motor[0],0] + G[motor[0],motor[1]] * np.cos(theta)
        x[motor[1], 1] = x[motor[0],1] + G[motor[0],motor[1]] * np.sin(theta)
        
        for i in np.where(visited_list)[0]:            
            active_list += list(self.find_unvisited_neighbors(i, C, visited_list))
            
        active_list = list(set(active_list))
        
        counter = 0
        
        while len(active_list)>0:
            k = active_list.pop(0)
            neighbors = self.find_neighbors(k, C)
            vn = neighbors[visited_list[neighbors]]
            if vn.shape[0]>1:
                if vn.shape[0]>2:
                    if show_msg:
                        print('Redudndant or overdefined system.')
                    return None, -2
                i = vn[0]
                j = vn[1]
                l_ij = np.linalg.norm(x[j]-x[i])
                s = np.sign((x0[i,1]-x0[k,1])*(x0[i,0]-x0[j,0]) - (x0[i,1]-x0[j,1])*(x0[i,0]-x0[k,0]))
                cosphi = (l_ij**2+G[i,k]**2-G[j,k]**2)/(2*l_ij*G[i,k])
                if cosphi >= -1.0 and cosphi <= 1.0:
                    phi = s * acos(cosphi)
                    R = np.array([[cos(phi), -sin(phi)],
                                  [sin(phi), cos(phi)]])
                    scaled_ij = (x[j]-x[i])/l_ij * G[i,k]
                    x[k] = np.matmul(R, scaled_ij.reshape(2,1)).flatten() + x[i]
                    path.append(k)
                    op.append([i,j,s])
                else:
                    if show_msg:
                        print('Locking or degenerate linkage!')
                    return None, -1
                
                visited_list[k] = True
                active_list += list(self.find_unvisited_neighbors(k, C, visited_list))
                active_list = list(set(active_list))
                counter = 0
            else:
                counter += 1
                active_list.append(k)
            
            if counter > len(active_list):
                if show_msg:
                    print('DOF larger than 1.')
                return None, 0
        return path, op
    
    def position_from_path(self, path, op, theta, x0, C, G, motor, fixed_nodes=[0, 1], show_msg=False):
        
        N = C.shape[0]
        x = np.zeros((N, 2))
        visited_list = np.zeros(N, dtype=bool)
        active_list = []
        
        visited_list[fixed_nodes] = True
        visited_list[motor[1]] = True
        
        x[fixed_nodes] = x0[fixed_nodes]
        
        dx = x0[motor[1],0] - x0[motor[0],0]
        dy = x0[motor[1],1] - x0[motor[0],1]
        
        theta_0 = np.math.atan2(dy,dx)
        theta = theta_0 + theta
        
        x[motor[1], 0] = x[motor[0],0] + G[motor[0],motor[1]] * np.cos(theta)
        x[motor[1], 1] = x[motor[0],1] + G[motor[0],motor[1]] * np.sin(theta)
        
        
        for l,step in enumerate(path):
            i = op[l][0]
            j = op[l][1]
            k = step
            
            l_ij = np.linalg.norm(x[j]-x[i])
            cosphi = (l_ij**2+G[i,k]**2-G[j,k]**2)/(2*l_ij*G[i,k])
            if cosphi >= -1.0 and cosphi <= 1.0:
                phi = op[l][2] * acos(cosphi)
                R = np.array([[cos(phi), -sin(phi)],
                              [sin(phi), cos(phi)]])
                scaled_ij = (x[j]-x[i])/l_ij * G[i,k]
                x[k] = np.matmul(R, scaled_ij.reshape(2,1)).flatten() + x[i]
            else:
                if show_msg:
                    print('Locking or degenerate linkage!')
                return np.abs(cosphi)
        return x
    
    def check(self, n_steps, x0, C, motor, fixed_nodes=[0, 1], show_msg=False, lim=[0.0,1.0]):
        G = self.get_G(x0,C)
        
        pop = self.get_path(x0, C, G, motor, fixed_nodes, show_msg)
        
        if not pop[0]:
            if pop[1] == 0:
                np.zeros([n_steps,C.shape[0],2]), False, True
            elif pop[1] == -1:
                np.zeros([n_steps,C.shape[0],2]), True, False
            elif pop[1] == -2:
                np.zeros([n_steps,C.shape[0],2]), False, True
        
        func = lambda t: self.position_from_path(pop[0],pop[1],t, x0, C, G, motor, fixed_nodes, show_msg)
        
        ts = np.linspace(0, 2*np.pi, n_steps)
        
        out = []
        
        for t in ts:
            x_temp = func(t)
            if np.array(x_temp).size == 1:
                return False
            else:
                out.append(x_temp)
            
        if np.max(out) <= lim[1] and np.min(out) >= lim[0]:
            return True
        else:
            return False
    
    def solve_rev(self, n_steps, x0, C, motor, fixed_nodes=[0, 1], show_msg=False):
        
        G = self.get_G(x0,C)
        
        pop = self.get_path(x0, C, G, motor, fixed_nodes, show_msg)

        if not pop[0]:
            return 10
        
        func = lambda t: self.position_from_path(pop[0],pop[1],t, x0, C, G, motor, fixed_nodes, show_msg)
        
        ts = np.linspace(0, 2*np.pi, n_steps)
        
        out = []
        
        for t in ts:
            x_temp = func(t)
            if np.array(x_temp).size == 1:
                if x_temp:
                    return np.zeros([n_steps,C.shape[0],2]), True, False
                else:
                    return np.zeros([n_steps,C.shape[0],2]), False, True
            else:
                out.append(x_temp)
        
        return np.array(out), False, False
    
    def material(self,x0, C):
        G = self.get_G(x0,C)
        return np.sum(G[np.logical_not(np.isinf(G))])    

def reduce(C,x,fixed_nodes,n):
    
    CC = C[:n+1,:n+1]
    xx = x[:n+1,:]
    fixed_nodess = fixed_nodes[fixed_nodes<=n]
    
    return CC,xx,fixed_nodess

def simplifier(C,x,fixed_nodes):
    
    ind = np.where(C.sum(0) == 0)[0]
    CC = np.delete(C,ind,0)
    CC = np.delete(CC,ind,1)
    xx = np.delete(x,ind,0)
    
    node_types = np.zeros([C.shape[0],1])
    node_types[fixed_nodes] = 1
    node_types = np.delete(node_types,ind,0)
    
    
    fixed_nodess = np.where(node_types == 1)[0]
    
    return CC,xx,fixed_nodess

def random_generator(g_prob = 0.15, n=None, N_min=8, N_max=20, pop = 50, strategy='srand'):
    
    if not n:
        n = int(np.round(np.random.uniform(low=N_min,high=N_max)))
    
    edges = [[0,2],[2,3],[1,3]]
    
    fixed_nodes = [0,1]
    motor = [0,2]
    
    node_types = np.random.binomial(1,g_prob,n-4)
    
    for i in range(4,n):
        if node_types[i-4] == 1:
            fixed_nodes.append(i)
        else:
            picks = np.random.choice(i,size=2,replace=False)
            
            while picks[0] in fixed_nodes and picks[1] in fixed_nodes:
                picks = np.random.choice(i,size=2,replace=False)
            
            edges.append([picks[0],i])
            edges.append([picks[1],i])
    
    C = np.zeros([n,n], dtype=bool)
    
    for edge in edges:
        C[edge[0],edge[1]] = True
        C[edge[1],edge[0]] = True
    
    
    fixed_nodes = np.array(fixed_nodes)
    
    solver = solver_cpu()
    
    if strategy == 'srand':
        x = np.random.uniform(low=0.0,high=1.0,size=[n,2])

        for i in range(4,n+1):

            sub_size = i
            invalid = not solver.check(100,x[0:sub_size],C[0:sub_size,0:sub_size],motor,fixed_nodes[np.where(fixed_nodes<sub_size)],False)

            co = 0

            while invalid:
                if sub_size > 4:
                    x[sub_size-1:sub_size] = np.random.uniform(low=0.0,high=1.0,size=[1,2])
                else:
                    x[0:sub_size] = np.random.uniform(low=0.0,high=1.0,size=[sub_size,2])

                invalid = not solver.check(50,x[0:sub_size],C[0:sub_size,0:sub_size],motor,fixed_nodes[np.where(fixed_nodes<sub_size)],False)

                co+=1

                if co == 100:
                    neighbours = np.where(C[sub_size-1])[0]
                    relavent_neighbours = neighbours[np.where(neighbours<sub_size-1)]
                    C[relavent_neighbours[0],sub_size-1] = 0
                    C[relavent_neighbours[1],sub_size-1] = 0
                    C[sub_size-1,relavent_neighbours[0]] = 0
                    C[sub_size-1,relavent_neighbours[1]] = 0
                    picks = np.random.choice(sub_size-1,size=2,replace=False)
                    C[picks[0],sub_size-1] = True
                    C[picks[1],sub_size-1] = True
                    C[sub_size-1,picks[0]] = True
                    C[sub_size-1,picks[1]] = True
                    co = 0
    else:
        co = 0
        x = np.random.uniform(low=0.05,high=0.95,size=[n,2])
        invalid = not solver.check(50,x,C,motor,fixed_nodes,False)
        while invalid:
            x = np.random.uniform(low=0.1+0.25*co/1000,high=0.9-0.25*co/1000,size=[n,2])
            invalid = not solver.check(50,x,C,motor,fixed_nodes,False)
            co += 1
            
            if co>=1000:
                return random_generator(g_prob, n, N_min, N_max, pop, strategy)
    return simplifier(C,x,fixed_nodes)

def random_generator_ns(g_prob = 0.15, n=None, N_min=8, N_max=20, pop = 50, strategy='srand'):
    
    if not n:
        n = int(np.round(np.random.uniform(low=N_min,high=N_max)))
    
    edges = [[0,2],[2,3],[1,3]]
    
    fixed_nodes = [0,1]
    motor = [0,2]
    
    node_types = np.random.binomial(1,g_prob,n-4)
    
    for i in range(4,n):
        if node_types[i-4] == 1:
            fixed_nodes.append(i)
        else:
            picks = np.random.choice(i,size=2,replace=False)
            
            while picks[0] in fixed_nodes and picks[1] in fixed_nodes:
                picks = np.random.choice(i,size=2,replace=False)
            
            edges.append([picks[0],i])
            edges.append([picks[1],i])
    
    C = np.zeros([n,n], dtype=bool)
    
    for edge in edges:
        C[edge[0],edge[1]] = True
        C[edge[1],edge[0]] = True
    
    
    fixed_nodes = np.array(fixed_nodes)
    
    solver = solver_cpu()
    
    if strategy == 'srand':
        x = np.random.uniform(low=0.0,high=1.0,size=[n,2])

        for i in range(4,n+1):

            sub_size = i
            invalid = not solver.check(100,x[0:sub_size],C[0:sub_size,0:sub_size],motor,fixed_nodes[np.where(fixed_nodes<sub_size)],False)

            co = 0

            while invalid:
                if sub_size > 4:
                    x[sub_size-1:sub_size] = np.random.uniform(low=0.0,high=1.0,size=[1,2])
                else:
                    x[0:sub_size] = np.random.uniform(low=0.0,high=1.0,size=[sub_size,2])

                invalid = not solver.check(50,x[0:sub_size],C[0:sub_size,0:sub_size],motor,fixed_nodes[np.where(fixed_nodes<sub_size)],False)

                co+=1

                if co == 100:
                    neighbours = np.where(C[sub_size-1])[0]
                    relavent_neighbours = neighbours[np.where(neighbours<sub_size-1)]
                    C[relavent_neighbours[0],sub_size-1] = 0
                    C[relavent_neighbours[1],sub_size-1] = 0
                    C[sub_size-1,relavent_neighbours[0]] = 0
                    C[sub_size-1,relavent_neighbours[1]] = 0
                    picks = np.random.choice(sub_size-1,size=2,replace=False)
                    C[picks[0],sub_size-1] = True
                    C[picks[1],sub_size-1] = True
                    C[sub_size-1,picks[0]] = True
                    C[sub_size-1,picks[1]] = True
                    co = 0
    else:
        co = 0
        x = np.random.uniform(low=0.05,high=0.95,size=[n,2])
        invalid = not solver.check(50,x,C,motor,fixed_nodes,False)
        while invalid:
            x = np.random.uniform(low=0.1+0.25*co/1000,high=0.9-0.25*co/1000,size=[n,2])
            invalid = not solver.check(50,x,C,motor,fixed_nodes,False)
            co += 1
            
            if co>=1000:
                return random_generator(g_prob, n, N_min, N_max, pop, strategy)
    return C,x,fixed_nodes

class curve_normalizer():
    def __init__(self, scale=True):
        
        
        self.scale = scale
        self.vfunc = np.vectorize(lambda c: self.get_oriented(c),signature='(n,m)->(n,m)')
        
    def get_oriented(self, curve):
        
        ci = 0
        t = curve.shape[0]
        pi = t
        
        while pi != ci:
            pi = t
            t = ci
            ci = np.argmax(np.linalg.norm(curve-curve[ci],2,1))
        
        d = curve[pi] - curve[t]
        
        if d[1] == 0:
            theta = 0
        else:
            d = d * np.sign(d[1])
            theta = -np.arctan(d[1]/d[0])
        
        rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        out = np.matmul(rot,curve.T).T
        out = out - np.min(out,0)
        
        if self.scale:
            out = out/np.max(out,0)[0]
        
        if np.isnan(np.sum(out)):
            out = np.zeros(out.shape)
                    
        return out
    
    def __call__(self, curves):
        return self.vfunc(curves)
    
class curve_normalizer_original_orientation():
    def __init__(self):
        self.vfunc = np.vectorize(lambda c: self.get_oriented(c),signature='(n,m)->(n,m)')
        
    def get_oriented(self, curve):
        xx = np.copy(curve)
        xx = xx - xx.min(0)
        d = xx.max(0)
        d = d[np.argmax(d)]
        xx = xx/d
        
        if np.isnan(np.sum(xx)):
            xx = np.zeros(xx.shape)
        
        return xx
    
    def __call__(self, curves):
        return self.vfunc(curves)
    
class gridizer():
    def __init__(self, res, d_range = [-1.5,1.5]):
        self.res = res
        self.vfunc = np.vectorize(self.gridize,signature='(n,m)->(k,k)')
        self.range = d_range
            
    def gridize(self, curve):
        if np.isnan(np.sum(curve)):
            return np.zeros([self.res,self.res]).astype(np.            bool)
        else:
            return rasterize_curve(curve,self.res)
    
    def __call__(self, curves):
        return self.vfunc(curves)

class gridizer_pc():
    def __init__(self, res, d_range = [-1.5,1.5]):
        self.res = res
        self.vfunc = np.vectorize(self.gridize,signature='(n,m)->(l,k)')
        self.range = d_range
            
    def gridize(self, curve):
        if np.isnan(np.sum(curve)):
            return np.array([])
        else:
            return rasterized_curve_coords(curve,self.res)
    
    def __call__(self, curves):
        out = []
        for i in range(curves.shape[0]):
            out.append(self.gridize(curves[i]).astype(np.int16))
        
        return out

def run_imap_multiprocessing(func, argument_list, show_prog = True):

    pool = mp.Pool(processes=mp.cpu_count())
    
    if show_prog:            
        result_list_tqdm = []
        for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list),position=0, leave=True):
            result_list_tqdm.append(result)
    else:
        result_list_tqdm = []
        for result in pool.imap(func=func, iterable=argument_list):
            result_list_tqdm.append(result)

    return result_list_tqdm

def output_heatmap(dummy):
    solver = solver_cpu()
    xs = []
    C,res,fixed_nodes = random_generator(g_prob=0.1,N_min=8,pop=10,strategy='rand')
    x = solver.solve_rev(50,res,C,[0,2,5],fixed_nodes,False)

    g = gridizer(250)
    
    return np.sum(g(np.swapaxes(x,0,1)),0)

def output_normalized_heatmap(dummy):
    solver = solver_cpu()
    
    xs = []
    C,res,fixed_nodes = random_generator(g_prob=0.1,N_min=8,pop=10,strategy='rand')
    x = solver.solve_rev(50,res,C,[0,2,5],fixed_nodes,False)

    g = gridizer(250,[0.,1.])
    n = curve_normalizer()
    return np.sum(g(n(np.swapaxes(x[:,3:,:],0,1))),0)>0            

def output_normalized_heatmap_ogo(dummy):
    solver = solver_cpu()
    
    xs = []
    C,res,fixed_nodes = random_generator(g_prob=0.1,N_min=8,pop=10,strategy='rand')
    x = solver.solve_rev(50,res,C,[0,2],fixed_nodes,False)

    g = gridizer(500,[0.,1.])
    n = curve_normalizer_original_orientation()
    return np.sum(g(n(np.swapaxes(x[:,3:,:],0,1))),0)>0

def get_random_curve(res=500, n = None):
    C,init_pos,fixed_nodes = random_generator(g_prob=0.1,n = n, N_min=6,N_max=20,pop=10,strategy='rand')
    
    solver = solver_cpu()
    g = gridizer(500,[0.,1.])
    n = curve_normalizer()
    
    x = solver.solve_rev(192,init_pos,C,[0,2],fixed_nodes,False)
    
    outs = g(n(np.swapaxes(x[:,3:,:],0,1)))
    
    ind = np.random.choice(outs.shape[0])
    
    while np.sum(outs[ind]) == 0.0:
         ind = np.random.choice(outs.shape[0])            
    return outs[ind], rasterized_curve_coords(n(np.swapaxes(x[:,3+ind:3+ind+1,:],0,1))[0],res)

def generate_dataset_entry(dummy, res = 500, solver_res = 192, g_prob=0.1,n = None, N_min=6,N_max=20,pop=10,strategy='rand'):
    
    np.random.seed()
    C,x0,fixed_nodes = random_generator(g_prob,n,N_min,N_max,pop,strategy)
    
    solver = solver_cpu()
    
    x = solver.solve_rev(solver_res,x0,C,[0,2],fixed_nodes,False)

    while np.array(x).size == 1:
        C,x0,fixed_nodes = random_generator(g_prob,n,N_min,N_max,pop,strategy)
        x = solver.solve_rev(solver_res,x0,C,[0,2],fixed_nodes,False)
        
    x = np.swapaxes(x,0,1)
    
    g = gridizer_pc(500,[0.,1.])
    n = curve_normalizer()
    nn = curve_normalizer_original_orientation()
    
    
    x_norm_ogo = nn(x)
    x_norm = n(x)
    x_point_cloud = g(x)
    x_point_cloud_norm_ogo = g(x_norm_ogo)
    x_point_cloud_norm = g(x_norm)
    
    return (C,x0,fixed_nodes), (x,x_norm_ogo,x_norm), (x_point_cloud,x_point_cloud_norm_ogo,x_point_cloud_norm)

def save_mechanism_json(C,x0,fixed_nodes,motor,save_name="mechanism.json", target=None):
    out = {"A": C.astype(np.int8).tolist(), "x0":x0.tolist(), "motor":motor, "fixed_nodes":fixed_nodes.tolist(),'target':target}
    
    with open(save_name, 'w') as outfile:
        json.dump(out, outfile)
        
def load_mechanism_json(file):
    f = open(file,)
    out = json.load(f)
    
    return out['A'],out['x0'],out['fixed_nodes']



class best(pymoo.util.display.Display):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        lowest = np.min(algorithm.pop.get("F"),0)
        for i in range(lowest.shape[0]):
            self.output.append("Lowest Memeber for Objective %i" % (i), lowest[i])
            
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def find_path(A, motor = [0,1], fixed_nodes=[0, 1]):
    
    # This function determines the path to the solution
    
    path = []
    
    A,fixed_nodes,motor = np.array(A),np.array(fixed_nodes),np.array(motor)
    
    unkowns = np.array(list(range(A.shape[0])))
    knowns = np.concatenate([fixed_nodes,[motor[-1]]])
    
    unkowns = unkowns[np.logical_not(np.isin(unkowns,knowns))]

    
    counter = 0
    while unkowns.shape[0] != 0:

        if counter == unkowns.shape[0]:
            # Non dyadic or DOF larger than 1
            return [], False
        n = unkowns[counter]
        ne = np.where(A[n])[0]
        
        kne = knowns[np.isin(knowns,ne)]
        
        if kne.shape[0] == 2:
            
            path.append([n,kne[0],kne[1]])
            counter = 0
            knowns = np.concatenate([knowns,[n]])
            unkowns = unkowns[unkowns!=n]
        elif kne.shape[0] > 2:
            #redundant or overconstraint
            return [], False
        else:
            counter += 1
    
    return np.array(path), True

def get_G(x0):
    return (np.linalg.norm(np.tile([x0],[x0.shape[0],1,1]) - np.tile(np.expand_dims(x0,1),[1,x0.shape[0],1]),axis=-1))

# @tf.function
# def get_GGpu(x0):
#     return (tf.norm(tf.tile([x0],[x0.shape[0],1,1]) - tf.tile(tf.expand_dims(x0,1),[1,x0.shape[0],1]),axis=-1))


def solve(path,x0,G,motor,fixed_nodes,theta):
    # Basic Solver Algorithm
    
    path,x0,G,motor,fixed_nodes = np.array(path),np.array(x0),np.array(G),np.array(motor),np.array(fixed_nodes)
    
    x = np.zeros_like(x0)
    
    x[fixed_nodes] = x0[fixed_nodes]
    x[motor[1]] = x[motor[0]] + G[motor[0],motor[1]] * np.array([np.cos(theta),np.sin(theta)])
    
    for step in path:
        i = step[1]
        j = step[2]
        k = step[0]
        
        l_ij = np.linalg.norm(x[j]-x[i])
        cosphi = (l_ij **2 + G[i,k]**2 - G[j,k]**2)/(2 * l_ij * G[i,k])
        if cosphi >= -1.0 and cosphi <= 1.0:
            s = np.sign((x0[i,1]-x0[k,1])*(x0[i,0]-x0[j,0]) - (x0[i,1]-x0[j,1])*(x0[i,0]-x0[k,0]))
            phi = s * np.math.acos(cosphi)
            R = np.array([[np.cos(phi), -np.sin(phi)],
                          [np.sin(phi), np.cos(phi)]])
            scaled_ij = (x[j]-x[i])/l_ij * G[i,k]
            x[k] = np.matmul(R, scaled_ij.reshape(2,1)).flatten() + x[i]
        else:
            #Locking or degenerate linkage!
            return x0, False, k
        
    return x, True, -1


def solve_full_revolution(path,A,x0,motor,fixed_nodes,n=200):
    # A function to solve for a full revolution of actuator Numerically
    
    path,x0,A,motor,fixed_nodes = np.array(path),np.array(x0),np.array(A),np.array(motor),np.array(fixed_nodes)
    
    thetas = np.linspace(0.0,2*np.pi,n)
    
    G = get_G(x0)
    
    solver = lambda t: solve(path,x0,G,motor,fixed_nodes,t)
    
    v_solve = np.vectorize(solver,signature='()->(n,2),(),()')
    
    return v_solve(thetas)

def solve_rev_vectorized(path,x0,G,motor,fixed_nodes,thetas):
    #Vectorized Faster Solver
    path,x0,G,motor,fixed_nodes = np.array(path),np.array(x0),np.array(G),np.array(motor),np.array(fixed_nodes)
    
    x = np.zeros([x0.shape[0],thetas.shape[0],2])
    
    x[fixed_nodes] = np.expand_dims(x0[fixed_nodes],1)
    x[motor[1]] = x[motor[0]] + G[motor[0],motor[1]] * np.concatenate([[np.cos(thetas)],[np.sin(thetas)]]).T
    
    state = np.zeros(thetas.shape[0])
    flag = True
    kk = np.zeros(thetas.shape[0]) - 1.0
    
    for step in path:
        i = step[1]
        j = step[2]
        k = step[0]
        
        l_ij = np.linalg.norm(x[j]-x[i],axis=-1)
        cosphi = (l_ij ** 2 + G[i,k]**2 - G[j,k]**2)/(2 * l_ij * G[i,k])

        state += np.logical_or(cosphi<-1.0,cosphi>1.0)
        
        kk = state * k * (kk==-1.0) + kk
        
        s = np.sign((x0[i,1]-x0[k,1])*(x0[i,0]-x0[j,0]) - (x0[i,1]-x0[j,1])*(x0[i,0]-x0[k,0]))

        phi = s * np.arccos(cosphi)

        a = np.concatenate([[np.cos(phi)],[-np.sin(phi)]]).T
        b = np.concatenate([[np.sin(phi)],[np.cos(phi)]]).T

        R = np.swapaxes(np.concatenate([[a],[b]]),0,1)

        scaled_ij = (x[j]-x[i])/np.expand_dims(l_ij,-1) * G[i,k]
        x[k] = np.squeeze(R @ np.expand_dims(scaled_ij,-1)) + x[i]
        
    kk = (kk!=-1.0) + kk
    
    return x, state == 0.0, kk.astype(np.int32)

def get_re_nodes(A,target):
    #This function reduced a mechanism to a given joint (Output is a list of joint indecies that are needed)

    neis = np.where(A[target])[0]
    neis = neis[neis<target]
    nodes = list(neis)
    if neis.shape[0] == 1:
        nodes += get_re_nodes(A,neis[0])
    elif neis.shape[0] == 2:
        nodes += get_re_nodes(A,neis[0]) + get_re_nodes(A,neis[1])
    
    return list(set(nodes + [target]))

def simulate_mechanism(M,n=200):
    #This function produces the data needed for the dataset
    A, x0, node_type = M
    
    fixed_nodes = np.where(node_type)[0]
    
    moving_nodes = np.where(node_type==0)[0]
    
    path = find_path(A,fixed_nodes=fixed_nodes)[0]
    
    x_sol = solve_rev_vectorized(path,x0,get_G(x0),[0,1],fixed_nodes,np.linspace(0,2*np.pi,n))[0]
    
    if np.isnan(x_sol).sum() != 0:
        return None
    
    x_sol_n = normalizer_object(x_sol)[moving_nodes]
    
    x_norm_cur = []
    x_norm_cur_i = []
    
    for i,mn in enumerate(moving_nodes):
        if np.isin(np.where(A[mn]),fixed_nodes).sum() != 0:
            if np.random.uniform()>=0.995:
                x_norm_cur.append(x_sol_n[i])
                x_norm_cur_i.append(mn)
        
        elif np.linalg.norm(x_sol_n[i] - [0.5,0.5],axis=-1).var()< 5e-4:
            if np.random.uniform()>=0.995:
                x_norm_cur.append(x_sol_n[i])
                x_norm_cur_i.append(mn)
        
        else:
            x_norm_cur.append(x_sol_n[i])
            x_norm_cur_i.append(mn)
            
    
    return x_sol, x_sol_n, moving_nodes, np.array(x_norm_cur), np.array(x_norm_cur_i)

def draw_mechanism(A,x0,fixed_nodes,motor, highlight=100, solve=True, thetas = np.linspace(0,np.pi*2,200), def_alpha = 1.0, h_alfa =1.0, h_c = "#f15a24"):
    
    #Draws a given mechanism

    def fetch_path():
        root = etree.parse(StringIO('<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 338"><defs><style>.cls-1{fill:#1a1a1a;stroke:#1a1a1a;stroke-linecap:round;stroke-miterlimit:10;stroke-width:20px;}</style></defs><path class="cls-1" d="M45.5,358.5l70.71-70.71M46,287.5H644m-507.61,71,70.72-70.71M223,358.5l70.71-70.71m20.18,70.72,70.71-70.71m13.67,70.7,70.71-70.71m20.19,70.72,70.71-70.71m15.84,70.71,70.71-70.71M345,39.62A121.38,121.38,0,1,1,223.62,161,121.38,121.38,0,0,1,345,39.62Z" transform="translate(-35.5 -29.62)"/></svg>')).getroot()
        view_box = root.attrib.get('viewBox')
        if view_box is not None:
            view_box = [int(x) for x in view_box.split()]
            xlim = (view_box[0], view_box[0] + view_box[2])
            ylim = (view_box[1] + view_box[3], view_box[1])
        else:
            xlim = (0, 500)
            ylim = (500, 0)
        path_elem = root.findall('.//{http://www.w3.org/2000/svg}path')[0]
        return xlim, ylim, parse_path(path_elem.attrib['d'])
    _,_,p = fetch_path()
    p.vertices -= p.vertices.mean(axis=0)
    p.vertices = (np.array([[np.cos(np.pi), -np.sin(np.pi)],[np.sin(np.pi), np.cos(np.pi)]])@p.vertices.T).T
    
    A,x0,fixed_nodes,motor = np.array(A),np.array(x0),np.array(fixed_nodes),np.array(motor)
    
    x = x0
    
    N = A.shape[0]
    for i in range(N):
        if i in fixed_nodes:
            if i == highlight:
                plt.scatter(x[i,0],x[i,1],color=h_c,s=700,zorder=10,marker=p)
            else:
                plt.scatter(x[i,0],x[i,1],color="#1a1a1a",s=700,zorder=10,marker=p)
        else:
            if i == highlight:
                plt.scatter(x[i,0],x[i,1],color=h_c,s=100,zorder=10,facecolors=h_c,alpha=0.7)
            else:
                plt.scatter(x[i,0],x[i,1],color="#1a1a1a",s=100,zorder=10,facecolors='#ffffff',alpha=0.7)
        
        for j in range(i+1,N):
            if A[i,j]:
                if (motor[0] == i and motor[1] == j) or(motor[0] == j and motor[1] == i):
                    plt.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#ffc800",linewidth=4.5)
                else:
                    plt.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#1a1a1a",linewidth=4.5,alpha=0.6)
                
    if solve:
        path = find_path(A,motor,fixed_nodes)[0]
        G = get_G(x0)
        x,c,k =  solve_rev_vectorized(path.astype(np.int32), x0, G, motor, fixed_nodes,thetas)
        x = np.swapaxes(x,0,1)
        if np.sum(c) == c.shape[0]:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        plt.plot(x[:,i,0],x[:,i,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                    else:
                        plt.plot(x[:,i,0],x[:,i,1],'--',color="#0078a7",linewidth=1.5, alpha=def_alpha)
        else:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        plt.plot(x[:,i,0],x[:,i,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                    else:
                        plt.plot(x[:,i,0],x[:,i,1],'--',color="#0078a7",linewidth=1.5, alpha=def_alpha)
            plt.text(0.5, 0.5, 'Locking Or Under Defined', color='red', horizontalalignment='center', verticalalignment='center')
        
    plt.axis('equal')
    plt.axis('off')
