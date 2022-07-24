class Simplex:
    def __init__(self, vertices, int_filt, cont_filt):
        self.verts     = vertices
        self.int_filt  = int_filt
        self.cont_filt = cont_filt


class Simplex0D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, coordinates, cc_list):
        super().__init__(vertices,int_filt,cont_filt)

        self.coords    = coordinates
        self.xcoord    = coordinates[0]
        self.ycoord    = coordinates[1]
        self.cc        = cc_list
        # List of 0-simplices in form of int_filt values

    # calc_cc works as follows:
    # We assume that it is calculated iteratively, so cc is calculated for timestep
    # before it is calculated for timestep+1.

    # If the timestep is in range(0,self.int_filt), then the vertex has not been born yet,
    # so its connected component is None.

    # If the timestep is in range(self_int_filt,len(S[0])), then the vertex has been born,
    # but not yet been merged with another vertex, so its connected component is self.int_filt.

    # If the timestep is larger then len(S[0]), then we need to figure out HOW MANY 1-SIMPLICES HAVE BEEN BORN SO FAR. ASSUME THIS NUMBER IS T.
    # We then look at the pivots from len(S[0]) to len(S[0])+T (recall that pivots
    # are ordered by the boundary matrix, and the boundary matrix is ordered first by
    # dimension, and only then by filtration.)
    # If the self.int_filt shows up in this sublist of relevant pivots, then we know
    # the vertex has been merged with another vertex, which has been born earlier.
    # We look at the 1-Simplex which has caused this merger, which is specified by the
    # index j of self.int_filt in the pivot list. We extract it by (S[1])[j - len(S[0])].
    # Once this edge is found, we look at its other vertex. This vertex need not be
    # born earlier than our original one, but it belongs to an earlier connected component.
    # Its own merger must have happened at least one timestep earlier.
    # Thus we take the cc of this other vertex from timestep-1 as the cc of self.


    # I changed this up so much that now I'm not even using pivots anymore?
    # How does this algorithm differ from UnionFind? I have not looked at it yet.
    def calc_cc(self,S,timestep,pivots):
        index = self.int_filt
        f_len = sum([len(S[i]) for i in range(4)])
        #print()
        #print("timestep",timestep)
        #print("index",index)
        # find out if new 1-simplex has been added at this timestep


        if timestep in range(index):
            #print("Case1")
            concom = None
        elif timestep in range(index,len(S[0])):
            #print("Case2")
            concom = index
        elif timestep in range(len(S[0]),f_len): # most interesting case
            #print("Case3")
            concom = self.cc[timestep-1]
            for i in range(len(S[1])):
                edge = None
                if ((S[1])[i]).int_filt == timestep: # at timestep we add a 1-simplex, creating the posibility of a merger
                    #print("Case3.1")
                # the 1-simplex has two vertices
                # first we calculate the new cc of these two vertices
                # then we look at the old cc's and at the cc of self
                # if our cc mages the old cc's, then it also gets updated
                    timestep_S0_S1 = i + len(S[0])
                    edge = (S[1])[i]
                    old_cc_0 = edge.vert0.cc[timestep-1]
                    old_cc_1 = edge.vert1.cc[timestep-1]
                    new_concom = min(old_cc_0,old_cc_1)
                    if self.cc[timestep-1] in [old_cc_0,old_cc_1]:
                        #print("new merger:",edge.verts)
                        concom = new_concom
                    else:
                        #print("no new merger")
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

        #print("connected component", concom)
        self.cc[timestep] = concom
        if len(self.cc) > f_len:
            self.cc = self.cc[:f_len]



class Simplex1D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, crossing_v, ordered_vertices):
        super().__init__(vertices, int_filt, cont_filt)

        self.cv = crossing_v
        self.ord_verts = ordered_vertices # list of Simplex0D elements
        self.vert0 = ordered_vertices[0]  # Simplex0D
        self.vert1 = ordered_vertices[1]  # Simplex0D

class Simplex2D(Simplex):
    def __init__(self, vertices, int_filt, cont_filt, ordered_vertices, boundary):
        super().__init__(vertices, int_filt, cont_filt)
        # boundary is a list of 3 objects of the class Simplex1D
        self.boundary = boundary

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
