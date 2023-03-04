# libraries
import matplotlib.pyplot as plt
import sympy as sp
from copy import deepcopy
from .. import lambda0 


# Creating Merge Tree

## MergeOrder class for ordering branches

class MergeOrder:
    def __init__(self, data):
        self.data_static = deepcopy(data)
        self.data = deepcopy(data)
        self.R = [] # list of ordered indices
        self.current_branch_index = 0
        self.current_branch = self.data[0]
        
    def set_branch(self, index):
        self.current_branch_index = index
        self.current_branch = self.data[self.current_branch_index]
        
    def generate_order(self):
        self.data = deepcopy(self.data_static)
        self.set_branch(0)
        
        self.iterator()
        return self.R
    
    def iterator(self):
        # current branch
        merger_list = self.current_branch[:-1]
        if len(merger_list) == 0:
            self.R.append(self.current_branch_index)
            #print("R:",self.R)
            
            if self.current_branch_index != 0:
                mother_index = self.current_branch[-1].new_component
                # reached end of branch, switch to next branch on mother branch {mother_index}
                self.data[mother_index].pop(-2)
                self.set_branch(mother_index)
                self.iterator()
        else:
            # go deeper
            new_branch_index = merger_list[-1].old_component
            self.set_branch(new_branch_index)
            self.iterator()

## Auxiliary functions

 
def colour_lookup():
    light_grey = '#e8e8e8' # dim = 0
    mid_grey   = '#bbbbbb' # dim = 1
    dark_grey  = '#5c5c5c' # dim = 2
    black      = '#000000' # dim = 3
    return {0: black,
            1: dark_grey,
            2: mid_grey,
            3: light_grey
           }
    

        
def extract_mergers_from_branch(branch):
    return [event for event in branch if isinstance(event, lambda0.Merger)]

def extract_sublattice_changes_from_branch(branch):
    return [event for event in branch if isinstance(event, lambda0.MonomialChange)]
 
def extract_mergers_global(data):
    mergers = [extract_mergers_from_branch(branch) for branch in data]
    last_event_time = mergers[0][-1].time_index
    mergers[0].append(lambda0.Merger(last_event_time + 2, 0, 0, None, None, None, None))
    return mergers

def extract_sublattice_changes_global(data):
    sublattice_changes = [extract_sublattice_changes_from_branch(branch) for branch in data]
    return sublattice_changes

def conditional_int2cont(time, continuous):
    if continuous is not None:
        time = continuous[time]
    return time


## Plotting single branches




def plot_branch(component, ax, merge_data, data, order, continuous):
    x1 = order.index(component)
    x2 = order.index(merge_data[component][-1].new_component)
    
    
    #if component == 0:
    #    age = data[component][-1].time_index + 1
    #else:
    #    age = merge_data[component][-1].time_index
    #    
    #age = conditional_int2cont(age, continuous)
    age = data[component][-1].time_index
    age = conditional_int2cont(age, continuous)
    if component == 0:
        age *= 1.1
    
    tick_length = 0.2
    tick_setoff = 0.05
    
    t0 = 0
    dim = 0
    det_rel = 1
    colour = colour_lookup()[0]
    
    ax.plot([x1+tick_setoff, x1+tick_setoff+tick_length], [0, 0], "black")
    ax.annotate(f"1.00 R^3", (x1 + 3/2*tick_length, 0))
    
    for event in data[component]:
        
        # annotations
        if ((event.det_rel != det_rel) or (event.dim != dim) 
            and isinstance(event, lambda0.MonomialChange)):
            event_time   = conditional_int2cont(event.time_index, continuous)
            # tick
            ax.plot([x1+tick_setoff, x1+tick_setoff+tick_length], [event_time, event_time], "black")
            
            # label
            if isinstance(event.det_rel, sp.core.mul.Mul):
                ax.annotate(f"{event.det_rel} R^{3-event.dim}", (x1 + 3/2*tick_length, event_time))
            else:
                ax.annotate(f"{float(event.det_rel):.2f} R^{3-event.dim}", (x1 + 3/2*tick_length, event_time))
            det_rel = event.det_rel
            

        # colour changes
        if (event.dim != dim):
            event_time = conditional_int2cont(event.time_index, continuous)
            ax.plot([x1, x1], [t0, event_time], colour)
            dim = event.dim
            colour = colour_lookup()[dim]
            t0  = event_time
        

    ax.plot([x1, x1], [t0, age], colour)
    if component == 0:
        ax.scatter(x1, age, c=colour, marker="^") # triangle to infinity
    else:
        ax.plot([x1, x2], [age, age], colour) # horizontal merge bar

        
## plot_mergetree() function


def plot_mergetree(data, continuous = None, width=5, height=7):
    fig, ax = plt.subplots()
    
    merge_data = extract_mergers_global(data)
    order = MergeOrder(merge_data).generate_order()
    N = len(order)
    
    fig.set_size_inches(width, height)
    
    # create plot with vertical lines of correct lengths
    if continuous is None:
        ax.set_ylabel("timesteps (integer)") 
    else:
        ax.set_ylabel("timesteps (continuous)")
    ax.set_xlabel("component")
    
    yticks = [0]
    yticks += [conditional_int2cont(data[component][i].time_index, continuous) 
               for component in range(0, N) 
               for i in range(len(data[component]))]
    
    
    ax.set_xticks([i for i in range(N)])
    ax.set_xticklabels([str(comp) for comp in order])
    ax.set_yticks(yticks)
    
    
    [ax.spines[edge].set_visible(False) for edge in ['top', 'bottom', 'right', 'left']]
    
    
    
    for component in range(N):
        plot_branch(component, ax, merge_data, data, order, continuous)
        

