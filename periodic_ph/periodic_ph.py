# libraries
import numpy as np
import torus_alpha_complex as tac



class ExamplePrint:
    def __init__(self, points, a=1, b=1, c=1):
        self.points = points
        self.scale_x = a
        self.scale_y = b
        self.scale_z = c
        self.trafo_matrix = np.diag([a,b,c])
        self.N = len(points)
        
        self.groups = None
        self.pairs  = None
        self.evolution = None
        self.int2cont = None
        
    def calculate_pph(self):
        torus_filtration, simplex_objects = tac.torus_filtration(self.points,
                                                      a=self.scale_x,
                                                      b=self.scale_y,
                                                      c=self.scale_z)
        self.N = len(simplex_objects[0])
        self.int2cont = int2cont(torus_filtration)
        self.groups, self.pairs = PH(torus_filtration, simplex_objects)
        self.evolution = Lambda_0_evolution(torus_filtration, self.N, self.pairs, self.trafo_matrix);

        
    def describe_evolution(self):
        #plot_persistence_pairs(persistence_pairs)
        for comp in range(self.N):
            print("-------------------")
            print(f"{bold0}COMPONENT {comp}{bold1}")
            self.describe_component_evolution(comp)
            
            

    def describe_component_evolution(self, comp):
        for i in range(len(self.evolution[comp])):
            state = self.evolution[comp][i]
            if isinstance(state, StaticSublattice):
                print(f"""
{bold0}Timestep {state.time_index} ({self.int2cont[state.time_index]:2.4f}){bold1}
dimension = {state.dim}
determinant = {float(state.det_abs):2.2f}
det. ratio  = {float(state.det_rel):2.2f}
basis = \n{state.basis_matrix}
                """)
                
            elif isinstance(state, Merger):
                print(f"""
{bold0}Timestep {state.time_index} ({self.int2cont[state.time_index]:2.4f}){bold1}
Component {state.old_component} merged to component {state.new_component}.
""")
   

    def plot_evolution(self, cont_timesteps = True, width=5, height=7):
        if cont_timesteps:
            plot_mergetree(self.evolution, 
                           continuous = self.int2cont, 
                           width=width, 
                           height=height)
        else:
            plot_mergetree(self.evolution, 
                           continuous = None, 
                           width=width, 
                           height=height)