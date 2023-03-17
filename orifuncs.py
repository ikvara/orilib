import math
import numpy as np
import matplotlib.pyplot as plt

########################################################################
#conversions between orientation descriptors 
########################################################################
def eu2quat(phi1,Phi,phi2): 
    '''
    Euler angles to quaternion.
    '''
    q = np.zeros((4))
    q[0]=  np.cos(Phi/2)*np.cos((phi1+phi2)/2)
    q[1]= -np.sin(Phi/2)*np.cos((phi1-phi2)/2)
    q[2]= -np.sin(Phi/2)*np.sin((phi1-phi2)/2)
    q[3]= -np.cos(Phi/2)*np.sin((phi1+phi2)/2)
    
    if q[0]<0: q= -q
    
    return(q)


def quat2eu(q):
    '''
    Quaternion to Euler angles.
    '''
    chi = np.sqrt((q[0]**2+q[3]**2)*(q[1]**2+q[2]**2))
    
    if chi!=0:
        sin2 = -(-q[0]*q[2] - q[1]*q[3])
        cos2 = (-q[0]*q[1] + q[2]*q[3])
        
        sin1 = (-q[0]*q[2] + q[1]*q[3])
        cos1 = (-q[0]*q[1] - q[2]*q[3])

        phi1 = np.arctan2(sin1,cos1)
        Phi = np.arccos(q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2)
        phi2 = np.arctan2(sin2,cos2)
        
    else:
        if q[1]==0 and q[2]==0:
            phi1 = np.arctan2(-2*q[0]*q[3], q[0]**2-q[3]**2)
            Phi  = 0
            phi2 = 0
            
        elif q[0]==0 and q[3]==0:
            phi1 = np.arctan2(2*q[1]*q[2], q[1]**2-q[2]**2)
            Phi  = math.pi
            phi2 = 0
            
    phi1 %= 2*math.pi
    phi2 %= 2*math.pi

    return phi1, Phi, phi2
 

def eu2mat(phi1,Phi,phi2):
    '''
    Euler angles to orientation matrix.
    '''
    g11 = np.cos(phi1)*np.cos(phi2) - np.sin(phi1)*np.sin(phi2)*np.cos(Phi)
    g12 = np.sin(phi1)*np.cos(phi2) + np.cos(phi1)*np.sin(phi2)*np.cos(Phi)
    g13 = np.sin(phi2)*np.sin(Phi)
    g21 = -np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(Phi)
    g22 = -np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(Phi)
    g23 = np.cos(phi2)*np.sin(Phi)
    g31 = np.sin(phi1)*np.sin(Phi)
    g32 = -np.cos(phi1)*np.sin(Phi)
    g33 = np.cos(Phi)
    
    matrix = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
    
    return matrix

def mat2eu(M):
    '''
    Orientation matrix to Euler angles.
    '''
    C = M[2,2]    
    Phi = np.arccos(C)
    
    if C==1:
        phi1 = np.arccos(M[0,0])
        phi2 = 0
        
    elif C==-1:
        phi1 = np.arccos(M[0,0])
        phi2 = math.pi
        
    else:
        phi1 = np.arctan2(M[2,0], -M[2,1])
        phi2 = np.arctan2(M[0,2], M[1,2])
        
    return phi1%(2*math.pi),Phi,phi2%(2*math.pi)

def mat2quat(matrix):
    '''
    Orientation matrix to quaternion.
    '''
    q0 = np.sqrt(1 + matrix[0,0] + matrix[1,1] + matrix[2,2])
    q1 = np.sqrt(1 + matrix[0,0] - matrix[1,1] - matrix[2,2])
    q2 = np.sqrt(1 - matrix[0,0] + matrix[1,1] - matrix[2,2])
    q3 = np.sqrt(1 - matrix[0,0] - matrix[1,1] + matrix[2,2])
    
    if matrix[2,1]<matrix[1,2]: q1 = -q1
    if matrix[0,2]<matrix[2,0]: q2 = -q2
    if matrix[1,0]<matrix[0,1]: q3 = -q3
    
    q = 1./2* np.array([q0,q1,q2,q3])
    q /= np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    
    return q  

def quat2mat(q): 
    '''
    Quaternion to orientation matrix.
    '''
    g11 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    g12 = 2*(q[1]*q[2] - q[0]*q[3])
    g13 = 2*(q[1]*q[3] + q[0]*q[2])
    g21 = 2*(q[1]*q[2] + q[0]*q[3])
    g22 = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    g23 = 2*(q[2]*q[3] - q[0]*q[1])
    g31 = 2*(q[1]*q[3] - q[0]*q[2])
    g32 = 2*(q[2]*q[3] + q[0]*q[1])
    g33 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    
    matrix = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
    
    return matrix

def mat2axisangle(matrix):
    '''
    Orientation matrix to axis-angle pair.
    '''
    arg = (matrix[0,0] + matrix[1,1] + matrix[2,2] -1)/2
    if arg>1:
        arg = 1
    elif arg<-1:
        arg = -1
    angle = np.arccos(arg)
    s = np.sin(angle)
    
    if angle!=0 and angle!=np.pi:
        r1 = (matrix[2,1]-matrix[1,2])/(2*s)
        r2 = (matrix[0,2]-matrix[2,0])/(2*s)
        r3 = (matrix[1,0]-matrix[0,1])/(2*s)
        axis = np.array([r1,r2,r3])
    elif angle==0:
        axis = np.array([0,0,1])
    else:
        r1 = np.sqrt((matrix[0,0]+1)/2)
        r2 = np.sqrt((matrix[1,1]+1)/2)
        r3 = np.sqrt((matrix[2,2]+1)/2)
        axis = np.array([r1,r2,r3])
 
    return axis, angle

    

def axisangle2mat(axis, angle): 
    '''
    Axis-angle pair to orientation matrix.
    '''
    r1 = axis[0]
    r2 = axis[1]
    r3 = axis[2]
    
    c = np.cos(angle)
    s = np.sin(angle)
    
    g11 = (1-c)*r1**2 + c
    g12 = r1*r2*(1-c) - r3*s
    g13 = r1*r3*(1-c) + r2*s
    g21 = r1*r2*(1-c) + r3*s
    g22 = (1-c)*r2**2 + c
    g23 = r2*r3*(1-c) - r1*s
    g31 = r1*r3*(1-c) - r2*s
    g32 = r2*r3*(1-c) + r1*s
    g33 = (1-c)*r3**2 + c
    
    matrix = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
    
    return matrix


def axisangle2quat(axis, angle): 
    '''
    Axis-angle pair to quaternion.
    '''
    q0 = np.cos(angle/2)
    q1 = axis[0]*np.sin(angle/2)
    q2 = axis[1]*np.sin(angle/2)
    q3 = axis[2]*np.sin(angle/2)
    
    return np.array([q0,q1,q2,q3])

def quat2axisangle(q): 
    '''
    Quaternion to axis-angle pair.
    '''
    angle = 2* np.arccos(q[0])
    
    if angle==0:
        axis = [0,0,1]
                
    else:
        s = 1./np.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
        axis = [s*q[1], s*q[2], s*q[3]]
    
    return axis, angle

def rodr2axisangle(r): 
    '''
    Rodrigues vector to axis-angle pair.
    '''
    tan = np.sqrt(r[0]**2 + r[1]**2 +r[2]**2)
    
    angle = 2*np.arctan(tan)
    r1 = r[0]/tan
    r2 = r[1]/tan
    r3 = r[2]/tan
    
    axis = np.array([r1,r2,r3])
    
    return axis, angle

def axisangle2rodr(axis, angle): 
    '''
    Axis-angle pair to Rodrigues vector.
    '''
    tan = np.tan(angle/2)
    
    r1 = tan*axis[0]
    r2 = tan*axis[1]
    r3 = tan*axis[2]
    
    return np.array([r1,r2,r3]) 

def mat2rodr(matrix):
    '''
    Orientation matrix to Rodrigues vector.
    '''
    axis, angle = mat2axisangle(matrix)
    
    return axisangle2rodr(axis, angle)

def quat2rodr(q): 
    '''
    Quaternion to Rodrigues vector. 
    '''
    r1 = q[1]/q[0]
    r2 = q[2]/q[0]
    r3 = q[3]/q[0]
        
    return np.array([r1,r2,r3])


########################################################################
# 
########################################################################
def product(p,q):
    '''
    Computes the product of two quaternions -- there's maybe a built-in function for it.
    '''
    return([p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3],
            p[0]*q[1]+p[1]*q[0]+p[2]*q[3]-p[3]*q[2],
            p[0]*q[2]+p[2]*q[0]-p[1]*q[3]+p[3]*q[1],
            p[0]*q[3]+p[3]*q[0]+p[1]*q[2]-p[2]*q[1]])

def symetries_quats(): 
    '''
    Symmetry quaternions for cubic system.
    '''
    symQ=[]
    symQ.append(np.array([1,0,0,0]))
    symQ.append(np.array([0,1,0,0]))
    symQ.append(np.array([0,0,1,0]))
    symQ.append(np.array([0,0,0,1]))
    
    symQ.append(np.array([0.5,0.5,0.5,0.5]))
    symQ.append(np.array([0.5,-0.5,-0.5,-0.5]))
    symQ.append(np.array([0.5,0.5,-0.5,0.5]))
    symQ.append(np.array([0.5,-0.5,0.5,-0.5]))
    
    symQ.append(np.array([0.5,-0.5,0.5,0.5]))
    symQ.append(np.array([0.5,0.5,-0.5,-0.5]))
    symQ.append(np.array([0.5,-0.5,-0.5,0.5]))
    symQ.append(np.array([0.5,0.5,0.5,-0.5]))
    
    symQ.append(np.array([1/np.sqrt(2),1/np.sqrt(2),0,0]))
    symQ.append(np.array([1/np.sqrt(2),0,1/np.sqrt(2),0]))
    symQ.append(np.array([1/np.sqrt(2),0,0,1/np.sqrt(2)]))
    symQ.append(np.array([1/np.sqrt(2),-1/np.sqrt(2),0,0]))
    
    symQ.append(np.array([1/np.sqrt(2),0,-1/np.sqrt(2),0]))
    symQ.append(np.array([1/np.sqrt(2),0,0,-1/np.sqrt(2)]))
    symQ.append(np.array([0,1/np.sqrt(2),1/np.sqrt(2),0]))
    symQ.append(np.array([0,-1/np.sqrt(2),1/np.sqrt(2),0]))
    
    symQ.append(np.array([0,0,1/np.sqrt(2),1/np.sqrt(2)]))
    symQ.append(np.array([0,0,-1/np.sqrt(2),1/np.sqrt(2)]))
    symQ.append(np.array([0,1/np.sqrt(2),0,1/np.sqrt(2)]))
    symQ.append(np.array([0,-1/np.sqrt(2),0,1/np.sqrt(2)]))
    
    return symQ

def symetries_matrix(): 
    '''
    Symmetry matrices for cubic system.
    '''
    matrices = []
    matrices.append(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    matrices.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    matrices.append(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
    matrices.append(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
    
    matrices.append(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    matrices.append(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    matrices.append(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
    matrices.append(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))
    
    matrices.append(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]))
    matrices.append(np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]))
    matrices.append(np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])) 
    matrices.append(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
    
    matrices.append(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    matrices.append(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    matrices.append(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])) 
    matrices.append(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    matrices.append(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))
    matrices.append(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    matrices.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))
    matrices.append(np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]))    
    
    matrices.append(np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    matrices.append(np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]))
    matrices.append(np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]))
    matrices.append(np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]))
    
    return matrices

def mis_q(p,q): 
    '''
    Computes the misorientation angle of two orientations represented by quaternions.
    '''
    pinv = np.array([p[0], -p[1], -p[2], -p[3]]) #inversion quaternion
    
    pq = product(pinv,q)     
    if pq[0]>1: pq[0] = 1 
    if pq[0]<-1: pq[0] = -1
    if pq[0]<0: pq[0] = abs(pq[0])
    
    angle = 2*np.arccos(pq[0])
    
    return angle

def mis_m(M1,M2): 
    '''
    Computes the misorientation angle of two orientations represented by orientation matrices.
    '''
    M1inv = np.linalg.inv(M1) #inversion matrix
    Mmis = M1inv.dot(M2)
    
    # Mmis = M1.dot(np.linalg.inv(M2))
    
    _, angle = mat2axisangle(Mmis)
    
    return angle


def dis_q(q1, q2):
    '''
    Computes the disorientation angle of two orientations represented by quaternions.
    '''
    alpha = 5
    
    symQ = symetries_quats()
    for sym in symQ:
        q2sym=product(sym,q2)
     
        mis = mis_q(q1, q2sym)       
        if mis<alpha: alpha = mis
    
    return alpha


def dis_m(M1, M2): 
    '''
    Computes the disorientation angle of two orientations represented by orientation matrices.
    '''
    alpha = 5
    
    symM = symetries_matrix()
    for sym in symM:
        M2sym=sym.dot(M2)
                      
        mis = mis_m(M1, M2sym)        
        if mis<alpha: alpha = mis
    
    return alpha


########################################################################
# Inner product (Arnold et al. 2018) -- for cubic symmetries
########################################################################
def fourth_product(u):
    '''
    Computes the symmetric 4-way array.
    '''
    prod = np.zeros((3,3,3,3))
    for j1 in range(3):
        for j2 in range(3):
            for j3 in range(3):
                for j4 in range(3):
                    prod[j1, j2, j3, j4] = u[j1]*u[j2]*u[j3]*u[j4]
    return prod

def symm_prod_I3():
    symm = np.zeros((3,3,3,3))
    for j1 in range(3):
        for j2 in range(3):
            for j3 in range(3):
                for j4 in range(3):
                    symm[j1, j2, j3, j4] = 1./3* ( ((j1==j2)*(j3==j4)) + ((j1==j3)*(j2==j4)) + ((j1==j4)*(j2==j3)) )
    return symm

def t_inner_prod(u, v):
    '''
    Computes the inner product of t(X) and t(Y).
    '''
    prod = 0
    for i in range(3):
        for j in range(3):
            prod += (u[i][0]*v[j][0] + u[i][1]*v[j][1] + u[i][2]*v[j][2])**4
    return prod - 9./5
    

def t_norm(u):
    '''
    Computer the norm of t(X), where X is the orientation matrix.
    '''
    prod = t_inner_prod(u, u)
    return np.sqrt(prod)


def t_functional(matrix):
    '''
    Returns the symmetric array corresponding to X (still cubic symmetries).
    '''
    t = np.zeros((3,3,3,3))   
    
    u1 = [matrix[0,0], matrix[0,1], matrix[0,2]]
    u2 = [matrix[1,0], matrix[1,1], matrix[1,2]]
    u3 = [matrix[2,0], matrix[2,1], matrix[2,2]]
    
    t = fourth_product(u1) + fourth_product(u2) + fourth_product(u3) - 3./5*symm_prod_I3()
    
    return t

def prod_fun(t1, t2):
    '''
    Computes the inner product of t1 and t2, which are outputs of the function t_functional().
    '''
    prod = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    prod += t1[i,j,k,l]*t2[i,j,k,l]
    return prod


########################################################################
# Fundamental zone 
########################################################################
def is_fz(q):
    '''
    Checks whether the orientation represented by quaternion lies in the chosen fundamental zone.
    '''
    _, Phi, phi2 = quat2eu(q)
    
    c2 = np.cos(phi2)
    s2 = np.sin(phi2)
                    
    minPhi = np.arccos( min((c2/np.sqrt(1+c2**2)), (s2/np.sqrt(1+s2**2)) ) )
    
    Bool = False
    if (Phi < math.pi/2) and (Phi >= minPhi):
       if phi2 < math.pi/2:
           Bool = True
           
    return Bool

def grid_tF(step_len=0.05):
    '''
    Grid the fundamental zone
    '''
    # first divide phi1, cos(Phi), phi2 intervals, i.e., [0, pi/2], [0,sqrt(2)/3] , [0,2pi]
    phis1 = np.arange(0, 2*math.pi, step_len)
    cs = np.arange(0, np.sqrt(1/3), step_len)
    phis2 = np.arange(0, math.pi/2, step_len)
    
    grid = []
    for p1 in phis1:
        for c in cs:
            for p2 in phis2:
                c2 = np.cos(p2)
                s2 = np.sin(p2)    

                max_c = min((c2/np.sqrt(1+c2**2)), (s2/np.sqrt(1+s2**2)))                
                if c <= max_c:
                    grid.append([p1, c, p2])
        
    return grid

def plot_transf_fz(orientations, int_nr, PATH, representation='quaternions', save=True):
    '''
    Plots the (transverse) cross-sections of the fundamental zone represented by Euler angles.

    orientations ... list of orientations
    int_nr ... number of cross-sections
    PATH ... Path, where you want to save the figures
    representation ... how are the orientations represented (either 'quaternions', 'matrices' or 'eu'). Default value is 'quaternions'
    '''
    if representation=='quaternions':
        quats = orientations
        
    elif representation=='matrices':
        quats = []
        for M in orientations:
            quats.append(mat2quat(M))
        
    elif representation=='eu':
        quats = []
        for eus in orientations:
            quats.append(eu2quat(eus[0], eus[1], eus[2]))
        
    sections = []
    for _ in range(int_nr):
        sections.append([])
    phis1 = np.linspace(0, 2*math.pi, num=int_nr+1)
    
    symQ = symetries_quats()
    for quat in quats:
        i = 0
        sym = symQ[i]
        qsym = product(sym,quat)
        qsym = np.asanyarray(qsym)
                
        while is_fz(qsym)==False:
            i += 1
            sym = symQ[i]
            qsym = product(sym,quat)
            qsym = np.asanyarray(qsym)
            
            if qsym[0]<0 : qsym = -qsym
            
        phi1, Phi, phi2 = quat2eu(qsym)
        
        idx = 0
        while (phi1>phis1[idx+1]) and (idx<int_nr):
            idx += 1
            
        sections[idx].append([phi2, np.cos(Phi), phi1])
            
    
    for i in range(int_nr):
        section = sections[i]
        
        phis2 = [line[0] for line in section]
        Phis = [line[1] for line in section]
        # obsphis1 = [line[2] for line in section]
        
        fig = plt.figure(dpi=300)
        ax = plt.gca()
        plt.scatter(phis2, Phis, s=5, alpha=1, c='dimgray')
        plt.xlabel('$\\varphi_2$', fontsize=20)
        plt.ylabel('$\\cos \\phi$', fontsize=20)
        ax.set_xlim(0, math.pi/2)
        ax.set_xticks([0, math.pi/2])
        ax.set_xticklabels([0, '$\\pi/2$'])
        ax.set_ylim(0,0.6)
        ax.set_yticks([0, np.sqrt(3)/3])
        ax.set_yticklabels([0, '$1/\\sqrt{3}$'])
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        plt.title(f'$\\varphi_1 \in$ ({round(phis1[i], 2)},{round(phis1[i+1], 2)})', fontsize=20)
        plt.tight_layout()

        if save:
            plt.close()
            
            fig.savefig(PATH+'eus_'+str(int_nr)+'_crosssections_'+str(i)+'.png', transparent=False)


########################################################################
#  Orientation characteristics
########################################################################
def normvec(v):
    return np.sqrt(v[0]**2+v[1]**2+v[2]**2)

def tilt(G, v):
    symetries = symetries_matrix()

    u = np.array([0,0,1])
    cos = -2
    for S in symetries:
        Msym = S.dot(G)
        
        vnew = Msym.dot(v)
        c = np.arccos((u.dot(vnew))/(normvec(u)*normvec(vnew)))
    
        if c>cos:
            cos = c

    return cos