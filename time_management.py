import math
from numba import jit
from sympy.solvers import solve
from sympy import symbols, Eq

from mathfunc import gompertz 

# t=f(tGA) relationship from T.Tallinen
@jit
def GA_to_t_TTallinen(tga):

    return 6.926*10**(-5)*tga**3 - 0.00665*tga**2 + 0.250*tga - 3.0189 # [X.Wang et al. 2019]

@jit
def t_to_GA_TTallinen(t):
    tga = symbols('x')
    equation = Eq(6.926*10**(-5)*tga**3 - 0.00665*tga**2 + 0.250*tga - 3.0189 - t)
    solution = solve(equation) #tga

    return solution

# Armstrong-GI-data-based t=f(tGA) relationship [Armstrong et al. 1995]
@jit
def GA_to_t_XWang(tga):
    
    return gompertz(tga, 0.987, 0.134, 29.433)

@jit
def t_to_GA_XWang(t):
    
    return 29.433 - 1/0.134*math.log(-math.log(t/0.987)) 