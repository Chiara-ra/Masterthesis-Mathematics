# contains plotting for merge trees 




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
        #print()
        #print("current branch", self.current_branch_index)
        merger_list = self.current_branch[:-1]
        if len(merger_list) == 0:
            self.R.append(self.current_branch_index)
            #print("R:",self.R)
            
            if self.current_branch_index != 0:
                mother_index = self.current_branch[-1].new_component
                #print(f"reached end of branch, switch to next branch on mother branch {mother_index}")
                self.data[mother_index].pop(-2)
                self.set_branch(mother_index)
                self.iterator()
        else:
            #print("go deeper")
            new_branch_index = merger_list[-1].old_component
            #print("new branch index",new_branch_index)
            self.set_branch(new_branch_index)
            self.iterator()









def plot_mergetree(data):
    fig, ax = plt.subplots()
    order = MergeOrder(data).generate_order()
    N = len(order)
    
    # create plot with vertical lines of correct lengths
    ax.set_ylabel("time steps (integer)") # for now everything in integer time steps
    ax.set_xlabel("component") 
    ax.set_xticks([i for i in range(N)])
    ax.set_xticklabels([str(comp) for comp in order])
    
    [ax.spines[edge].set_visible(False) for edge in ['top', 'bottom', 'right', 'left']]
    for component in range(N):
        x1 = order.index(component)
        x2 = order.index(data[component][-1].new_component)
        age = data[component][-1].time
        ax.plot([x1, x1], [0, age], "black")
        ax.plot([x1, x2], [age, age], "black")

        
        
        
def extract_mergers_from_branch(branch):
    return [event for event in branch if type(event) == Merger]
 
def extract_mergers_global(data):
    mergers = [extract_mergers_from_branch(branch) for branch in data]
    last_event_time = mergers[0][-1].time
    mergers[0].append(Merger([True, last_event_time + 2, 0, 0]))
    return mergers