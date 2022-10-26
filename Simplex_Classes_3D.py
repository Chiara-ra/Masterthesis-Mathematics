# As a note for future versions:
#!!!!!
#!!!!!
# It would probably make a lot of sense to add the attributes .index_all and .index_dim to any given simplex.
# So a given simplex would have its int_filt, which tells us its place in the filtration.
# It would also have index_all, which tells us its place in the list of all simplices,
# first ordered by dimension, then by filtration.
# The value index_dim would then only tell us its place in the list of dim-simplices,
# ordered by their filtration values.


import numpy as np





class Simplex:
    def __init__(self, vertices, int_filt, cont_filt):
        self.verts     = vertices
        self.index_total = int_filt # used to be int_filt
        self.index_dim   = int_filt
        self.time = cont_filt # used to be cont_filt

    def update_int_filt(self, new_val):
        self.index_total = new_val

        
        
        
        
class Simplex0D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, coordinates):
        super().__init__(vertices,int_filt,cont_filt)

        self.coords    = coordinates
        self.xcoord    = self.coords[0]
        self.ycoord    = self.coords[1]
        
        # self.cc is List of 0-simplices in form of int_filt values
        # Make .cc a method, not something that is initialised right at the beginning


    def calc_cc(self, S, timestep):
        index = self.int_filt
        f_len = sum([len(S[i]) for i in range(4)])
        # find out if new 1-simplex has been added at this timestep
        """
        can we simplify this so that we only look at the death-times of cc's, not at all timesteps
        """

        if timestep in range(index):
            con_comp = None
        elif timestep in range(index,len(S[0])):
            con_comp = index
        elif timestep in range(len(S[0]),f_len): # most interesting case
            con_comp = self.cc[timestep-1]
            for i in range(len(S[1])):
                edge = None
                if ((S[1])[i]).int_filt == timestep: 
                    # at timestep we add a 1-simplex, creating the posibility of a merger
                    
                    # the 1-simplex has two vertices
                    # first we calculate the new cc of these two vertices
                    # then we look at the old cc's and at the cc of self
                    # if our cc mages the old cc's, then it also gets updated
                    timestep_S0_S1 = i + len(S[0])
                    edge = (S[1])[i]
                    old_cc_0 = edge.vert0.cc[timestep-1]
                    old_cc_1 = edge.vert1.cc[timestep-1]
                    new_con_comp = min(old_cc_0,old_cc_1)
                    if self.cc[timestep-1] in [old_cc_0,old_cc_1]:
                        # new merger:
                        con_comp = new_con_comp
                    else:
                        # no new merger
                        break
                elif ((S[1])[i]).int_filt > timestep: # no 1-simplex is added
                    #print("Case3.2")
                    break



        elif timestep >= f_len:
            #print("Case4")
            raise ValueError("timestep exceeds number of filtration steps")
        else:
            #print("Case5")
            raise ValueError("timestep is not valid.")

        self.cc[timestep] = con_comp
        if len(self.cc) > f_len:
            self.cc = self.cc[:f_len]

class Simplex1D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, crossing_v, ordered_vertices):
        super().__init__(vertices, int_filt, cont_filt)

        self.cv = crossing_v.astype(np.int32)
        self.ord_verts = ordered_vertices # list of Simplex0D elements
        self.vert0 = ordered_vertices[0]  # Simplex0D
        self.vert1 = ordered_vertices[1]  # Simplex0D

class Simplex2D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, ordered_vertices, boundary):
        super().__init__(vertices, int_filt, cont_filt)
        self.ord_verts = ordered_vertices
        self.boundary = boundary # boundary is a list of 3 objects of the class Simplex1D
        
        
        
class Simplex3D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, ordered_vertices, boundary):
        super().__init__(vertices, int_filt, cont_filt)
        self.ord_verts = ordered_vertices
        self.boundary = boundary # boundary is a list of 4 objects of the class Simplex2D       
        


class Persistence_Pair:
    def __init__(self, dimension, interval_int, interval_cont, rep_simplices):
        self.dim        = dimension
        self.pair_int   = interval_int # life span of class in integer steps
        self.start_int  = interval_int[0]
        self.end_int    = interval_int[1]
        self.pair_cont  = interval_cont # life span of class in real time
        self.start_cont = interval_cont[0]
        self.end_cont   = interval_cont[1]
        self.reps       = rep_simplices # list of Simplex-objects whose sum gives representative
        self.cc         = None # we first have to call self.calc_cc
        #self.cv         = None # we first have to call self.calc_cc

    def calc_cc(self): # right now this only works for 1-dim pps. Generalise this for 0 to 3. 
        first_rep = self.reps[0]
        first_vert = first_rep.vert0
        cc = first_vert.cc
        self.cc = cc

    def return_cv(self, timestep):
        V = np.array([0,0],dtype=np.int32)
        a = self.start_int

        if timestep >= a:
            edge = self.reps[0] # class Simplex1D
            V += edge.cv      # add crossing vector of edge to V
            end_vert = edge.vert1
            pp_red = self.reps[1:]

            while len(pp_red) != 0:
                # look for correct representative that has end_vert as one of its vertices
                for i in range(len(pp_red)):
                    edge = pp_red[i]

                    # check if we are looking at the correct edge
                    if end_vert in edge.ord_verts:
                        # check if orientations are lining up or not
                        if end_vert == edge.ord_verts[0]:
                            ori = +1
                            end_vert = edge.vert1
                        else:
                            ori = -1
                            end_vert = edge.vert0

                        V += ori * pp_red[i].cv
                        pp_red = pp_red[:i]+pp_red[i+1:]

                        break # this breaks the for loop, but the while loop continues

                    if i == len(pp_red)-1:
                        print("Could not find next edge.") # make this real error message
                        pp_red = []
                        break

        else:
            raise ValueError("Timestep is earlier than birth of cycle.")

        return V
