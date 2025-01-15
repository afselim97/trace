#%%
import numpy as np
import itertools
from tqdm import tqdm
from typing import List,Iterator
from numpy.typing import NDArray
import random
from numba import njit
from numba.typed import List as numba_List

#%%
class trace():
    """
    An object used to calculate pairwise coalescent rates as a function of time
    """
    def __init__(
        self,
        tree_sequences: Iterator,
        L_list: List[int],
        n_samples: int
        ) -> None:
        """instantiation iterates over an iterator of tree sequences. For each tree sequence, it reads off L local trees, as specified by the L_list parameter.
            For each of these trees, a preorder traversal of the local tree is stored in a column of the "traversal_matrix." The time of these nodes and the time of their parents is also stored in other matrices.

        Args:
            tree_sequences ([Iterator[tskit.TreeSequence]): either an iterator of tree sequence is provided, or a single tree sequence. If an iterator is provided, then a list is expected for the L_list parameter. If a single tree sequence is provided, hten a single integer value is expected for L_trees (number of trees to extract from this tree sequence)
            L_list (List[int]): A list for the number of trees to be sampled from each tree sequence.
            n_samples[int] : number of samples in the trees
        """
        # Basic definitions
        self.n = n_samples
        self.sample_ids_list = np.arange(n_samples)
        self.L = np.sum(L_list) # Total number of trees

        burnin = 0.1 # This is the percentage of a chromosome in the beginning and the end to avoid extracting trees from (telomere and centromere)
        cumsum_L = np.concatenate(([0],np.cumsum(L_list)))  

        traversal_matrix = np.zeros((2*self.n-1,self.L),int) # A matrix that will store preorer traversals of the nodes in each tree
        node_time_matrix = np.zeros((2*self.n-1,self.L)) # A matrix that will store the times of each node in each tree, organized by their pre-order traversal
        parent_time_matrix = np.zeros((2*self.n-1,self.L)) # A matrix that will store the times of the parent each node in each tree, organized by their pre-order traversal
        num_nodes_tree = np.zeros((self.L),int) # Number of nodes in each tree
        
        # extracting the data
        for i,ts in tqdm(enumerate(tree_sequences)):
            L=L_list[i]
            if L == 0: ## Case where no trees are to be extracted from this tree sequence
                continue

            num_trees = ts.num_trees
            node_times = ts.nodes_time
            trees_iter = ts.trees()

            if L == 1 and ts.num_trees == 1: # Case where the tree sequence has a single tree (simulating a chromosome of length 1)
                trees = trees_iter 
            else: 
                try:
                    print(f"Extracting {L} trees from {num_trees} total trees")
                    trees = itertools.islice(trees_iter,int(burnin*num_trees),int((1-burnin)*num_trees-((1-2*burnin)*num_trees)%L),int(((1-2*burnin)*num_trees)/L)) # Extracting L equally spaced trees from the tree sequence, ignoring the beginning and the end of the chromosome
                except: 
                    trees_iter = ts.trees()
                    print(f"extracting {L} random trees")
                    inxs = np.sort(random.sample(range(num_trees),L))
                    trees = []
                    for inx in inxs:
                        trees.append(next(itertools.islice(trees_iter, inx, inx+1)))
                        
            for j,tree in enumerate(trees): # Populating the matrices defined above
                inx = cumsum_L[i]+j
                preorder = tree.preorder()
                n_nodes = num_nodes_tree[inx] = len(preorder)
                traversal_matrix[:n_nodes,inx] = preorder
                node_time_matrix[:n_nodes,inx] = node_times[preorder]
                parent_time_matrix[:n_nodes,inx] = node_times[tree.parent_array[preorder]]
                
        parent_time_matrix[0,:] = np.max(node_time_matrix[0,:]) # Setting the parent of each local root to the global mrca   
        
        # Defining the basic attributes of the class

        self.traversal_matrix = traversal_matrix
        self.node_time_matrix = node_time_matrix
        self.parent_time_matrix = parent_time_matrix
        self.num_nodes_tree = num_nodes_tree
        self.nodes = traversal_matrix[:,0][~np.in1d(traversal_matrix[:,0],self.sample_ids_list)]

    def coal_events_tree(self,preorder_traversal: List[int], node_times: List[float], t_list_lower: List[float], t_list_upper: List[float], window_list_upper: List[float]) -> NDArray[np.bool_]:
        """For a single tree represented by its preorder traversal, and a sliding time window with potentially variable width,  finds time points where each pair of samples coalesced.

        Args:
            preorder_traversal (List[int]): The preorder traversal of the nodes of the tree
            node_times (List[float]): The time of these nodes, ordered by their preorder traversal
            t_list_lower (List[float]): The lower bound of each time window
            t_list_upper (List[float]): The lower bound of each subsequent window
            window_list_upper (List[float]): The upper bound of each time window

        Returns:
            NDArray[np.bool_]: a 3d array, where (i,j,k) is true if samples i and j coalesced between the lower bound of window k-1 and the lower bound of window k
            NDArray[np.bool_]: a 3d array, where (i,j,k) is true if samples i and j coalesced within window k. Note that the windows can be overlapping.
        """
        coalesced_between_windows = np.zeros((self.n,self.n,len(t_list_lower)),bool)
        coalesced_within_windows = np.zeros((self.n,self.n,len(t_list_lower)),bool)

        for node in self.nodes: # Each node represents a coalescence event
            leaf_sets = get_leaf_sets(preorder_traversal,node,self.sample_ids_list) # At each node, 2 or more clades coalesce, such that all samples between each pair of clades coalesce at this time
            node_time = node_times[preorder_traversal==node][0] # The time of this coalescent event
            above_lower_bound = node_time > t_list_lower # Event happened at a time above the lower bound of the time window
            below_coal_upper_bound = node_time < t_list_upper # The coal upper bound is basically the lower bound of the next time window
            below_window_upper_bound = node_time < window_list_upper # Event happened at a time below the upper bound of the time window

            coal_indices = np.where(np.logical_and(above_lower_bound,below_coal_upper_bound))[0] # A time index k where the coalecent event occured between t_list_lower[k] and t_list_lower[k+1]
            coalesced_within_windows_indices = np.where(np.logical_and(above_lower_bound,below_window_upper_bound))[0] # Indices of time windows where the coal event occured

            # Updating the boolean matrices with the coalescent information for a single node
            coalesced_between_windows = fill_matrix(coalesced_between_windows,leaf_sets,coal_indices) 
            coalesced_within_windows = fill_matrix(coalesced_within_windows,leaf_sets,coalesced_within_windows_indices)

        return coalesced_between_windows,coalesced_within_windows

    def compute_coal_rates(self,t_list: List[float], delta_list: List[float]) -> NDArray:
        """Computes time-varying coalescent rates between each two pairs of samples using a Nelson Aalen Estimate

        Args:
            t_list (List[float]): The list of time points that define the top bound of each time window. The first time point will always start from zero.
            delta_list (List[float]): A list of window sizes

        Returns:
            NDArray[int]: Number of uncoalesced trees between each two samples as a function of time
            NDArray[int]: Number of trees where a coalescence event occured between each two samples within each time window
            NDArray[int]: Pairwise coalescent rate as a function of time.
        """
        delta_list = np.array(delta_list)

        traversal_matrix = self.traversal_matrix
        nodes_time_matrix = self. node_time_matrix
        
        if t_list[0] == 0:
            t_list_lower = t_list
            window_list_upper = t_list_lower + delta_list
            t_list_upper = np.concatenate((t_list[1:],[window_list_upper[-1]]))
        else:
            t_list_lower = np.concatenate(([0],t_list))
            delta_list = np.concatenate(([delta_list[0]],delta_list))
            window_list_upper = t_list_lower + delta_list
            t_list_upper = np.concatenate((t_list,[window_list_upper[-1]]))

        num_coal_events_between_windows = np.zeros((self.n,self.n,len(t_list_lower)),np.uint64)
        num_coal_events_within_windows = np.zeros((self.n,self.n,len(t_list_lower)),np.uint64)

        for i in tqdm(range(self.L)):
            traversal = traversal_matrix[:,i]
            times = nodes_time_matrix[:,i]
            coalesced_between_windows,coalesced_within_windows = self.coal_events_tree(traversal,times,t_list_lower,t_list_upper,window_list_upper)
            num_coal_events_between_windows[coalesced_between_windows] += 1
            num_coal_events_within_windows[coalesced_within_windows] += 1

        num_coal_events_between_windows[np.arange(self.n),np.arange(self.n),:] = 0
        num_coalesced_through_time = np.cumsum(num_coal_events_between_windows,axis=2)
        num_coalesced_through_time = np.dstack((np.zeros((self.n,self.n)),num_coalesced_through_time)) # Adding the zero time points
        num_uncoalesced_through_time = self.L - num_coalesced_through_time
        num_uncoalesced_through_time = num_uncoalesced_through_time[:,:,:-1]

        rates_through_time = num_coal_events_within_windows / (delta_list*num_uncoalesced_through_time+1e-9) ## Nelson Aalen estimate of pairwise coalescent rates
        rates_through_time[np.arange(self.n),np.arange(self.n),:] = 0

        if t_list[0] !=0:
            num_uncoalesced_through_time = num_uncoalesced_through_time[:,:,1:]
            num_coal_events_within_windows = num_coal_events_within_windows[:,:,1:]
            rates_through_time = rates_through_time[:,:,1:]
        
        

        return num_uncoalesced_through_time,num_coal_events_within_windows,rates_through_time
    

@njit
def fill_matrix(matrix: NDArray[np.bool_], leaf_sets: List[List[int]], indices: List[int]) -> NDArray:
    num_sets = len(leaf_sets)
    for i in range(num_sets):
        for j in range(i+1,num_sets):
            for sample_1 in leaf_sets[i]:
                for sample_2 in leaf_sets[j]:
                    matrix[sample_1, sample_2, indices] = matrix[sample_2, sample_1, indices] = True
    return matrix

@njit()
def get_leaf_sets(preorder_traversal: NDArray[np.int64],parent_node_id: int,samples: NDArray[np.int64]) -> List[List[int]]:
    n=len(samples)
    leaf_sets = numba_List()
    num_nodes = len(preorder_traversal)
    parent_inx = np.where(preorder_traversal==parent_node_id)[0][0]
    if parent_inx == num_nodes-1:
        return leaf_sets
    else:
        child = preorder_traversal[parent_inx+1]
    leaf_set = np.ones(n,dtype=np.int32)*-1
    i=0
    for k,node_id in enumerate(preorder_traversal[parent_inx+1:]):
        if child > parent_node_id: # Reached the end of the children (no more leaf sets for that node)
            break

        if np.any(samples == node_id):
            leaf_set[i] = node_id
            i+=1

        if parent_inx+k+2 == num_nodes: # Reached the end of the array
            remaining_set = leaf_set[leaf_set!=-1]
            if len(remaining_set) > 0:
                leaf_sets.append(remaining_set)
            break

        if preorder_traversal[parent_inx+2+k] > child: # Check if I need to start a new leaf set
            leaf_sets.append(leaf_set[leaf_set!=-1])
            child = preorder_traversal[parent_inx+2+k]
            i=0
            leaf_set = np.ones(n,dtype=np.int32)*-1
    
    return leaf_sets
# %%
