import os
home = os.getcwd()
import numpy as np
from numpy import pi as pi
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import sys
import subprocess
from sklearn.decomposition import PCA
import warnings
import math
import pymesh
import pylab as plt
import shutil
import csv
from pathlib import Path
path_to_repo = '/Users/Ali/repos/aliPyGBe'

'''
Taken from https://sites.google.com/site/dlampetest/python/triangulating-a-sphere-recursively 
Really appreciate you putting this code out there, thanks!
'''
'''
Create a unitsphere recursively by subdividing all triangles in an octahedron recursivly.

A unitsphere has a radius of 1, which also means that all points in this sphere
have an absolute value of 1. Another feature of an unitsphere is that the normals 
of this sphere are exactly the same as the vertices.

This recursive method will avoid the common problem of the polar singularity, 
produced by 2d parameterization methods.

If you wish a sphere with another radius than that of 1, simply multiply every single
value in the vertex array with this new radius 
(although this will break the "vertex array equal to normal array" property)
'''
import numpy
#import pylab

octahedron_vertices = numpy.array( [ 
    [ 1.0, 0.0, 0.0], # 0 
    [-1.0, 0.0, 0.0], # 1
    [ 0.0, 1.0, 0.0], # 2 
    [ 0.0,-1.0, 0.0], # 3
    [ 0.0, 0.0, 1.0], # 4 
    [ 0.0, 0.0,-1.0]  # 5                                
    ] )
octahedron_triangles = numpy.array( [ 
    [ 0, 4, 2 ],
    [ 2, 4, 1 ],
    [ 1, 4, 3 ],
    [ 3, 4, 0 ],
    [ 0, 2, 5 ],
    [ 2, 1, 5 ],
    [ 1, 3, 5 ],
    [ 3, 0, 5 ]] )

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n, 3) '''
    lens = numpy.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def divide_all( vertices, triangles ):    
    #new_triangles = []
    new_triangle_count = len( triangles ) * 4
    # Subdivide each triangle in the old approximation and normalize
    #  the new points thus generated to lie on the surface of the unit
    #  sphere.
    # Each input triangle with vertices labelled [0, 1,2] as shown
    #  below will be turned into four new triangles:
    #
    #            Make new points
    #                 a = (0+2)/2
    #                 b = (0+1)/2
    #                 c = (1+2)/2
    #        1
    #       /\        Normalize a, b, c
    #      /  \
    #    b/____\ c    Construct new triangles
    #    /\    /\       t1 [0, b,a]
    #   /  \  /  \      t2 [b, 1,c]
    #  /____\/____\     t3 [a, b,c]
    # 0      a     2    t4 [a, c,2]    
    v0 = vertices[ triangles[:,0] ]
    v1 = vertices[ triangles[:,1] ]
    v2 = vertices[ triangles[:,2] ]
    a = ( v0+v2 ) * 0.5
    b = ( v0+v1 ) * 0.5
    c = ( v1+v2 ) * 0.5  
    normalize_v3( a )
    normalize_v3( b )
    normalize_v3( c )
    
    #Stack the triangles together.
    vertices = numpy.hstack( (v0, b,a,  b, v1, c,  a, b,c, a, c,v2) ).reshape((-1, 3))
    
    #Now our vertices are duplicated, and thus our triangle structure are unnecesarry.    
    return vertices, numpy.arange( len(vertices) ).reshape( (-1, 3) )

def create_unit_sphere( recursion_level=2 ):
    vertex_array, index_array = octahedron_vertices, octahedron_triangles
    for i in range( recursion_level - 1 ):
        vertex_array, index_array  = divide_all(vertex_array, index_array)

    center = numpy.zeros((len(index_array), 3))
    for i in range(len(index_array)):
        triangle = numpy.array([vertex_array[index_array[i, 0]], vertex_array[index_array[i, 1]], vertex_array[index_array[i, 2]]])
        center[i,:] = numpy.dot(numpy.transpose(triangle), 1/3.*numpy.ones(3))
        
    return vertex_array, index_array, center


def vertex_array_only_unit_sphere( recursion_level=2 ):
    vertex_array, index_array = create_unit_sphere(recursion_level)
    if recursion_level > 1:    
        return vertex_array.reshape( (-1) )
    else:
        return vertex_array[index_array].reshape( (-1) )

def surfaceVariables(vertex, triangle):

    N = len(triangle)
    normal = numpy.zeros((N, 3))
    Area   = numpy.zeros(N)
    for i in range(N):
        y = vertex[triangle[i]]
        L = numpy.array([y[1]-y[0], y[2]-y[1], y[0]-y[2]])
        normal[i,:] = numpy.cross(L[0],L[2])
        norm_normal = numpy.linalg.norm(normal[i,:])
        Area[i]     = norm_normal/2.
        normal[i,:] /= norm_normal
    
    return normal, Area

def get_pqr_from_file(*,file=None, files=None, pop_end = 0):
    
    if file and files:
        raise TypeError('Not both!')
    if not files: 
        files = [file]
    net_pqr=[]
    
    for i in range(len(files)):
        
        with open(files[i]) as f:

            pqr = [line.rstrip('\n').split() for line in f]
            
        f.close()
        for i in range(pop_end):
            pqr.pop()
        pqr_new = np.reshape(pqr, np.shape(pqr))[:,-5:].astype(float)
        net_pqr.append(pqr_new)
    pqr=pqr_new
    
    
    for j in range(len(net_pqr)-1):
        pqr=np.vstack((pqr, net_pqr[-j-2]))
    return pqr


def pqrdf_to_df(*,file=None, files=None, pop_end = 0):
    
    if file and files:
        raise TypeError('Not both!')
    if not files: 
        files = [file]
    net_pqr = [] 
    for i in range(len(files)):
        
        with open(files[i]) as f:

            pqr = [line.rstrip('\n').split() for line in f]
            
        f.close()
        for i in range(pop_end):
            pqr.pop()
        pqr_new = np.reshape(pqr, np.shape(pqr))
        net_pqr.append(pqr_new)
    pqr=pqr_new
    
    for j in range(len(net_pqr)-1):
        pqr=np.vstack((pqr, net_pqr[-j-2]))
    
    df=pd.DataFrame(pqr, columns=["Type","Atom_number","Atom_type","AA_type",\
                                       "Chain","X", "Y", "Z", "q", "r"])
    df[["Atom_number", "Chain","X", "Y", "Z", "q", "r"]] = df[["Atom_number", "Chain","X", "Y", "Z", "q", "r"]].apply(pd.to_numeric, errors='coerce')
    return df

def vert_to_df(vert_file):

    return pd.DataFrame(np.loadtxt(vert_file),columns=["X", "Y", "Z"]).apply(pd.to_numeric, errors='coerce')

def odd_it(n):
    
    assert n,"n must be greater than or equal to 0"
    if n:
        return n - ((n-1) % 2)
    
    return 1

def get_rot(vector_orig, vector_fin):
    
    """
    Calculates the rotation matrix required to rotate from one vector to another. This is used when we
    assign the new charge locations in a conformed geometry.
    """
    
    R = np.zeros((3, 3))
    
    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = math.acos(np.dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = np.cos(angle)
    sa = np.sin(angle)

    # Calculate the rotation matrix elements.
    R[0, 0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0, 1] = -z*sa + (1.0 - ca)*x*y
    R[0, 2] = y*sa + (1.0 - ca)*x*z
    R[1, 0] = z*sa+(1.0 - ca)*x*y
    R[1, 1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1, 2] = -x*sa+(1.0 - ca)*y*z
    R[2, 1] = x*sa+(1.0 - ca)*y*z
    R[2, 2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)
    
    return R

def get_cylinder_positions_and_normals(r, h, theta):
    
    """
    Returns the surface normals of a toroidal surface.
    """      
    
    x =  r*np.cos(theta)
    z =  r*np.sin(theta)
    y =  h
    
    # Tangent w/r/t the circle with bigger radius (r = a)
    n_z = np.sin(theta)
    n_x = np.cos(theta)
    n_y = 0
    
    
    return x, y,z, n_x, n_y, n_z

def insert(file_name, string):
    
    """
    Inserts "string" ahead of "original_file" and saves as "new_file_name"
    """
    pwd = os.getcwd()
    if os.path.exists(os.path.join(pwd, file_name)):
        
        with open(file_name,'r') as f1:

            with open('temp.txt','w') as f2: 

                f2.write(string)
                f2.write(f1.read())

        f2.close()
        f1.close()

        os.rename('temp.txt',file_name)
    else:
        return "file %s does not exist!" % file_name
    pass

"""def get_flat_membrane_info(l, h,charge_to_side_tol, hex_pack_dist):
    
    # horizontal meaning along x-axis and vertical meaning along y-axis
    assert ((l>2*charge_to_side_tol) and (h>2*charge_to_side_tol)), "%s or %s < %s: L and H must be greater than 2*(charge-to-side tolerance)"\
        %(l, h,charge_to_side_tol)
        
    charge_limit_hor = l-2*charge_to_side_tol
    charge_limit_ver = h-2*charge_to_side_tol
    n_half_seg_hor = int (2*charge_limit_hor/(hex_pack_dist*np.sqrt(3)))
    n_half_seg_hor = odd_it(n_half_seg_hor)
    n_seg_ver = int (charge_limit_ver/hex_pack_dist)
    l_new = n_half_seg_hor*hex_pack_dist*np.sqrt(3)/2+2*charge_to_side_tol
    h_new = n_seg_ver*hex_pack_dist+2*charge_to_side_tol
    return l_new, h_new, n_half_seg_hor, n_seg_ver"""
    

def get_flat_membrane_info(l, h,charge_to_side_tol, hex_pack_dist):
    
    # horizontal meaning along x-axis and vertical meaning along y-axis
    #assert ((l>2*charge_to_side_tol) and (h>2*charge_to_side_tol)), "%s or %s < %s: L and H must be greater than 2*(charge-to-side tolerance)"\
    #    %(l, h,charge_to_side_tol)
        
    n_half_seg_hor = int (2*l/(hex_pack_dist*np.sqrt(3)))
    n_half_seg_hor = odd_it(n_half_seg_hor)
    n_seg_ver = int (h/hex_pack_dist)
    
    return n_half_seg_hor, n_seg_ver

def get_stern_thickness(interafce_type, stern_layer_t):

    """Returns the stern thickness depending on the type of interface."""

    dictionary = {'diel':0, 'stern':stern_layer_t}

    return dictionary.get(interafce_type,'Not Found')

def get_pos_and_norm_cyl(r, h, theta):
    
    """
    Returns the surface normals of a toroidal surface.
    """      
    
    x =  r*np.cos(theta)
    z =  r*np.sin(theta)
    y =  h
    
    # Tangent w/r/t the circle with bigger radius (r = a)
    n_z = np.sin(theta)
    n_x = np.cos(theta)
    n_y = 0
    
    
    return x, y,z, n_x, n_y, n_z

    
def write_file_info(file_name, info):
    tmp=open(file_name, "w")
    tmp.write(info)
    tmp.close()
    pass

def trisurf_from_mesh(mesh_file):
    
    vert_file = "%s.vert" % mesh_file
    face_file = "%s.face" % mesh_file
    
    with open(vert_file) as f1:

        xyz = [line.rstrip('\n').split() for line in f1]

    f1.close()

    # Array structure of the pqr file
    xyzArr = np.reshape(xyz, np.shape(xyz))

    # Positions (vertices)
    v_x = xyzArr[:,0].astype(float).flatten()
    v_y = xyzArr[:,1].astype(float).flatten()
    v_z = xyzArr[:,2].astype(float).flatten()

    with open(face_file) as f:

        face = [line.rstrip('\n').split() for line in f]

    f.close()

    # Array structure of the pqr file
    face = np.reshape(face, np.shape(face))

    # Faces (triangles)
    f_x = face[:,0].astype(int)-1
    f_y = face[:,1].astype(int)-1
    f_z = face[:,2].astype(int)-1

    face = np.vstack([f_x, f_y, f_z]).T
    
    return v_x, v_y, v_z, face

def mesh_sphere(rec, r,x0, y0, z0, filename, charge):
    
    xc = np.array([x0, y0, z0])
    vertex, index, center = create_unit_sphere(rec)
    vertex *= r
    vertex += xc

    index += 1 # Agrees with msms format
    index_format = np.zeros_like(index)
    index_format[:,0] = index[:,0]
    index_format[:,1] = index[:,2]
    index_format[:,2] = index[:,1]

    # Check
    x_test = np.average(vertex[:,0])
    y_test = np.average(vertex[:,1])
    z_test = np.average(vertex[:,2])

    f = open('mol.pqr',"w+")
    f.write("%s %5d %4s %5s %5d %9.6f %9.6f %9.6f %9.6f %9.6f\n" % ('ATOM',1,'NA','TMP',1, xc[0],xc[1],xc[2],charge, 1.0))
    f.close()

    if abs(x_test-x0)>1e-12 or abs(y_test-y0)>1e-12 or abs(z_test-z0)>1e-12:
        print ('Center is not right!')

    np.savetxt(filename+'.vert', vertex, fmt='%.4f')
    np.savetxt(filename+'.face', index_format, fmt='%i')

    print('Sphere with %i faces, radius %f and centered at %f,%f,%f was saved to the file '%(len(index), r, x0, y0, z0)+filename)
    pass


def get_vert_face_info(filename):
    
    with open('%s.vert'%filename) as f1:

        vert = [line.rstrip('\n').split() for line in f1]

    f1.close()
    with open('%s.face'%filename) as f2:

        face = [line.rstrip('\n').split() for line in f2]

    f2.close()
    vert_data = np.reshape(vert, np.shape(vert)).astype(float)
    face_data = np.reshape(face, np.shape(face)).astype(int)-1
    
    return vert_data, face_data

def fix_mesh(mesh, my_target_len, detail="normal"):
    import pymesh
    import numpy as np
    
    bbox_min, bbox_max = mesh.bbox;
    diag_len = np.linalg.norm(bbox_max - bbox_min);
    if detail == "normal":
        target_len = diag_len * 5e-3;
    elif detail == "high":
        target_len = diag_len * 2.5e-3;
    elif detail == "low":
        target_len = diag_len * 1e-2;
    print("Target resolution: {} mm".format(target_len));
    print("My Target resolution: {} mm".format(my_target_len));
    target_len = my_target_len

    count = 0;
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100);
    mesh, __ = pymesh.split_long_edges(mesh, target_len);
    num_vertices = mesh.num_vertices;
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6);
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                preserve_feature=True);
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100);
        if mesh.num_vertices == num_vertices:
            break;

        num_vertices = mesh.num_vertices;
        print("#v: {}".format(num_vertices));
        count += 1;
        if count > 10: break;

    mesh = pymesh.resolve_self_intersection(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh = pymesh.compute_outer_hull(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5);
    mesh, __ = pymesh.remove_isolated_vertices(mesh);

    return mesh;

def merge_geometries_from_file(filename_1, filename_2, iters=2, save_stl=True, output_filename='merged_Stern'):
    
    vert_face_data_1 = get_vert_face_info(filename_1)
    vert_face_data_2 = get_vert_face_info(filename_2)
    mesh1 = pymesh.form_mesh(vert_face_data_1[0],vert_face_data_1[1])
    mesh2 = pymesh.form_mesh(vert_face_data_2[0],vert_face_data_2[1])
    pymesh.save_mesh('%s.stl'% filename_1, mesh1)
    pymesh.save_mesh('%s.stl'% filename_2, mesh2)
    merged = pymesh.boolean(mesh1, mesh2, operation="union", engine = "cork")
    merged, __ = pymesh.remove_duplicated_faces(merged)
    
    for i in range(iters):
        #merged = pymesh.resolve_self_intersection(merged);
        merged, __ = pymesh.remove_degenerated_triangles(merged, 10)
        merged, __ = pymesh.collapse_short_edges(merged, rel_threshold=0.65 , preserve_feature=False) ##### New
        #merged, __ = pymesh.remove_obtuse_triangles(merged, max_angle=150, max_iterations=10)
        merged, __  = pymesh.remove_isolated_vertices(merged)
        merged, __ = pymesh.remove_obtuse_triangles(merged, max_angle=120, max_iterations=10)
        merged = pymesh.compute_outer_hull(merged, engine='auto')
        
    #merged = fix_mesh(merged, my_target_len)
    #merged, __ = pymesh.collapse_short_edges(merged, abs_threshold=0.25 , preserve_feature=False) ##### New
    faces = np.copy(merged.elements)
    faces+=1
    np.savetxt("%s.vert" % output_filename, merged.vertices, fmt='%5.3f',delimiter=' ')
    np.savetxt("%s.face" % output_filename, faces, fmt='%5.0f',delimiter=' ')
    
    if save_stl:
        
        pymesh.save_mesh('%s.stl' % output_filename, merged)

    pass

def meshRefiner(vert, face, tarPts, rad, level, subDiv, radReductionFactor=0.5):
    
    import scipy.interpolate as interp
    x = vert[:,0]
    y = vert[:,1]
    target_x = tarPts[:,0]
    target_y = tarPts[:,1]
    
    for lev in range(level):
        
        for j in range(len(target_x)):
        #for j in range(9):

            refinedPts=[]
            unrefinedPts=[]
            for i in range(x.size):
                if ((x[i]-target_x[j])*(x[i]-target_x[j]) + (y[i]-target_y[j])*(y[i]-target_y[j]) < rad*rad):
                    refinedPts.append(i)
                else:
                    unrefinedPts.append(i)
            x_new = x[refinedPts]
            y_new = y[refinedPts]

            x_old = x[unrefinedPts]
            y_old = y[unrefinedPts]
            newTris=[]
            for i in range(face.shape[0]):
                if ((face[i][0] in refinedPts) and (face[i][1] in refinedPts) and (face[i][2] in refinedPts)):
                    newTris.append(face[i])
            triang = mtri.Triangulation(x_new, y_new)
            my_tri = delete_connectivity(triang)
            refiner = mtri.UniformTriRefiner(my_tri)
            my_tri2, index = refiner.refine_triangulation(subdiv=subDiv, return_tri_index=True)
            x = np.hstack((x_old, my_tri2.x))
            y = np.hstack((y_old, my_tri2.y))
            newTriangs  = mtri.Triangulation(x, y)
            face = newTriangs.triangles
        
        rad = radReductionFactor*rad
    
    
    x_init = vert[:,0]
    y_init = vert[:,1]
    z_init = vert[:,2]
    
    interpFunRbf = interp.Rbf(x_init, y_init, z_init, function='cubic', smooth=0)  # default smooth=0 for interpolation
    z = interpFunRbf(x, y)
    nodes = np.vstack((x, y,z)).T
    
    fig, ax1 = plt.subplots()
    
    ax1.set_aspect('equal')
    ax1.triplot(newTriangs, c='blue',linewidth=0.2)
    
    return nodes, face

def msms2stl(filename):     
    
    with open('%s.vert' % filename) as f_vert:
        lines_v = [line_v.rstrip('\n').split() for line_v in f_vert]
    f_vert.close()
    x_v = []
    y_v =[]
    z_v = []
    for line_v in lines_v:
        x_v.append(float(line_v[0]))
        y_v.append(float(line_v[1]))
        z_v.append(float(line_v[2]))

    vert_mol = np.array([[x_v[i],y_v[i],z_v[i]] for i in range(len(lines_v))], dtype=np.float32)
    with open('%s.face' % filename) as f_face:
        lines_f = [line_f.rstrip('\n').split() for line_f in f_face]
    f_face.close()
    x_f = []
    y_f =[]
    z_f = []
    for line_f in lines_f:
        x_f.append(int(line_f[0])-1)
        y_f.append(int(line_f[1])-1)
        z_f.append(int(line_f[2])-1)

    face_mol = np.array([[x_f[i],y_f[i],z_f[i]] for i in range(len(lines_f))], dtype=np.int)
    mesh = pymesh.form_mesh(vert_mol, face_mol)
    pymesh.save_mesh('%s.stl' % filename, mesh)
    
    return mesh

def stl2msms(mesh, file_name):
    
    """
    converts mesh object to msms vert and face files.
    """
    
    py_mesh = mesh
    py_mesh, __ = pymesh.remove_duplicated_vertices(py_mesh)
    py_mesh, __ = pymesh.remove_duplicated_faces(py_mesh)
    new_elements = np.copy(py_mesh.elements)
    
    
    ## PyMesh  numbers elements starting from 0
    for i in range(new_elements.shape[0]):
        
        for j in range(new_elements.shape[1]):
            
            new_elements[i, j]+=1
    
    np.savetxt("%s.vert" % file_name, py_mesh.vertices, fmt='%5.3f',delimiter=' ')
    np.savetxt("%s.face" % file_name, new_elements, fmt='%5.0f',delimiter=' ')
    pass

def gen_stern_xyzr(stern_radius, xyzr_diel='test.xyzr'):

    xyzr_stern = xyzr_diel + '.stern'

    X = np.loadtxt(xyzr_diel)

    X[:,3] += stern_radius

    np.savetxt(xyzr_stern, X, fmt='%5.5f')
    pass

def msms_salt_mol(i, salt, molXYZR = "test.xyzr",
                  path_to_nlbc_data ='/Users/Ali/repos/setschenow-data/nlbc_test',
                  msmsPath = '/Users/Ali/Downloads/msms_MacOSX_2.6.1/msms.MacOSX.2.6.1'):
    
    #dictionary = {'NaCl': 1.415,'NH4Cl': 1.590,'NaBr': 1.490,'Na2SO4': 1.587,\
    #              'LiCl': 1.200,'KCl': 1.595,'K2SO4': 1.827,'CaCl2': 1.493,'NH42SO4': 1.820}
    Joung_radii = {'LI' : 0.9430, 'NA' : 1.2595, 'K' : 1.5686, 'RB' : 1.6680, 'CS' : 1.8179, 'F' : 2.1188, 'CL' : 2.3120, 'BR' : 2.3994, 'I' : 2.6312}
    dictionary = {'NaCl': (Joung_radii.get('NA')+Joung_radii.get('CL'))/2,
                  'KCl': (Joung_radii.get('K')+Joung_radii.get('CL'))/2,
                  'NaBr': (Joung_radii.get('NA')+Joung_radii.get('BR'))/2,
                  'LiCl': (Joung_radii.get('LI')+Joung_radii.get('CL'))/2}
    #salt='NaCl'
    stern_radius = dictionary.get(salt,'Not Found')
    
    for molecule in get_setschenow_data(salt)[0]:
        
        path_to_mol_data = os.path.join(path_to_nlbc_data, molecule) 
        os.chdir(path_to_mol_data)
        salt_diel = "diel_%s_%s"%(molecule, salt)
        msms_diel = "%s%s%s%s%s%s%s" % (msmsPath,' -if ',molXYZR,' -no_header -de ',i,' -prob 1.4 -of ',salt_diel)
        gen_stern_xyzr(stern_radius)
        molXYZR_stern = molXYZR+'.stern'
        salt_stern = "stern_%s_%s"%(molecule, salt)
        msms_stern = "%s%s%s%s%s%s%s" % (msmsPath,' -if ',molXYZR_stern,' -no_header -de ',i,' -prob 1.4 -of ',salt_stern)
        os.system(msms_diel)
        os.system(msms_stern)
        
        mol_salt_srf = "test_setschenow_%s.srf" % salt
        Path('%s/%s' % (path_to_mol_data, salt_diel)).touch()
        Path('%s/%s' % (path_to_mol_data, salt_stern)).touch()
        f=open(mol_salt_srf,"w+")
        f.write("f\nf\n./%s\n./%s\n\n\n0\n" % (salt_stern, salt_diel))
        f.close()
        os.chdir(path_to_nlbc_data)
    pass

def msms_salt_ion(i, ions, salt, molXYZR = "test.xyzr",
                  path_to_nlbc_data ='/Users/Ali/repos/setschenow-data/nlbc_test',
                  msmsPath = '/Users/Ali/Downloads/msms_MacOSX_2.6.1/msms.MacOSX.2.6.1'):
    

    #dictionary = {'NaCl': 1.415,'NH4Cl': 1.590,'NaBr': 1.490,'Na2SO4': 1.587,\
    #              'LiCl': 1.200,'KCl': 1.595,'K2SO4': 1.827,'CaCl2': 1.493,'NH42SO4': 1.820}
    Joung_radii = {'Li' : 0.9430, 'Na' : 1.2595, 'K' : 1.5686, 'Rb' : 1.6680, 'Cs' : 1.8179, 'F' : 2.1188, 'Cl' : 2.3120, 'Br' : 2.3994, 'I' : 2.6312}
    dictionary = {'NaCl': (Joung_radii.get('Na')+Joung_radii.get('Cl'))/2,
                  'KCl': (Joung_radii.get('K')+Joung_radii.get('Cl'))/2,
                  'NaBr': (Joung_radii.get('Na')+Joung_radii.get('Br'))/2,
                  'LiCl': (Joung_radii.get('Li')+Joung_radii.get('Cl'))/2}
    #salt='NaCl'
    stern_radius = dictionary.get(salt,'Not Found')
    
    for ion in ions:
        
        path_to_mol_data = os.path.join(path_to_nlbc_data, ion) 
        os.chdir(path_to_mol_data)
        f = open(molXYZR,"w+")   
        r_ion = Joung_radii.get(ion,'Not Found')
        mol_D_xyzr = "%10.5f%10.5f%10.5f%10.5f\n" % (0, 0,0, r_ion) # Na-Na or Na-Na(-)
        #Cl_D_xyzr = "%8.5f%8.5f%8.5f%8.5f\n" % (-distance/2, 0,0, 2.312) # Cl-Cl or Na-Cl
        f.write(mol_D_xyzr)
        f.close()

        molXYZR_stern = molXYZR+'.stern'
        # Cl stern mesh
        f = open(molXYZR_stern,"w+")   
        mol_S_xyzr = "%10.5f%10.5f%10.5f%10.5f\n" % (0, 0,0, r_ion+stern_radius) # Na-Na or Na-Na(-)
        #Cl_S_xyzr = "%8.5f%8.5f%8.5f%8.5f\n" % (-distance/2, 0,0, 4.312) # Cl-Cl or Na-Cl
        f.write(mol_S_xyzr)
        f.close()
        salt_diel = "diel_%s_%s"%(ion, salt)
        msms_diel = "%s%s%s%s%s%s%s" % (msmsPath,' -if ',molXYZR,' -no_header -de ',i,' -prob 1.4 -of ',salt_diel)
        salt_stern = "stern_%s_%s"%(ion, salt)
        msms_stern = "%s%s%s%s%s%s%s" % (msmsPath,' -if ',molXYZR_stern,' -no_header -de ',i,' -prob 1.4 -of ',salt_stern)
        os.system(msms_diel)
        os.system(msms_stern)
        
        mol_salt_srf = "test_setschenow_%s.srf" % salt
        Path('%s/%s' % (path_to_mol_data, salt_diel)).touch()
        Path('%s/%s' % (path_to_mol_data, salt_stern)).touch()
        f=open(mol_salt_srf,"w+")
        f.write("f\nf\n./%s\n./%s\n\n\n0\n" % (salt_stern, salt_diel))
        f.close()
        os.chdir(path_to_nlbc_data)
    pass

def get_solid_angle(area, r_avg):
     
    return math.acos(1 - area/2/np.pi/(r_avg**2))
    

def get_setschenow_data(salt, setschenowPath = '/Users/Ali/repos/setschenow-data'):
    
    a=[]
    salt_csv_path = os.path.join(setschenowPath,'%s.csv'%salt)
    with open(salt_csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            a.append(row)
    data = np.reshape(a, np.shape(a))
    mols = data[1:,0]
    setschenow_coeffs = data[1:,1]
    
    return mols, setschenow_coeffs

def translate_mesh(stl_mesh_file, offset=[0, 0, 0],rotation_axis=[1, 0,0],rotation_angle=0,\
                   out_name='translated',write_stl=True, write_msh=True):
    
    mesh = pymesh.load_mesh("%s.stl"%stl_mesh_file)
    offset = np.array(offset)
    axis = np.array(rotation_axis)
    angle = math.radians(rotation_angle)
    rot = pymesh.Quaternion.fromAxisAngle(axis, angle)
    rot = rot.to_matrix()
    vertices = mesh.vertices
    bbox = mesh.bbox
    centroid = 0.5 * (bbox[0] + bbox[1])
    vertices = np.dot(rot, (vertices - centroid).T).T + centroid + offset
    translated_mesh = pymesh.form_mesh(vertices, mesh.faces)
    if write_stl:
        pymesh.save_mesh("%s_%s.stl" % (out_name, stl_mesh_file),translated_mesh)
    if write_msh:
        pymesh.save_mesh("%s_%s.msh" % (out_name, stl_mesh_file),translated_mesh)
    
    return translated_mesh

def generate_vert_face_from_msh(mesh_name, out_name, repo_path=path_to_repo):
    
    pwd = os.getcwd()
    gen_file = 'generate_vert_face.sh'
    if not os.path.exists(os.path.join(pwd, gen_file)):
        subprocess.Popen('cp %s/%s .' % (repo_path, gen_file), shell=True).wait()
    gen_cmd = './%s %s' % (gen_file, mesh_name)
    subprocess.Popen(gen_cmd, shell=True).wait()
    face_file = 'face.txt'
    vert_file = 'vert.txt'

    data_face = pd.read_csv(face_file, sep=" ", header=None)
    data_vert = pd.read_csv(vert_file, sep=" ", header=None)

    data_face.columns = ["el_num", "el_type", "reg_p", "reg_el","num_node",'p1','p2','p3'] 
    data_vert.columns = ['num','X','Y','Z'] 

    data_vert['X'] = data_vert['X'].map('{0:14.6f}'.format)
    data_vert['Y'] = data_vert['Y'].map('{0:12.6f}'.format)
    data_vert['Z'] = data_vert['Z'].map('{0:12.6f}'.format)

    data_face['X'] = data_face['p1'].map('{0:10d}'.format)
    data_face['Y'] = data_face['p2'].map('{0:10d}'.format)
    data_face['Z'] = data_face['p3'].map('{0:10d}'.format)

    np.savetxt("%s.vert" % out_name, data_vert[["X", "Y", "Z"]].values, fmt='%s')
    np.savetxt("%s.face" % out_name, data_face[["p1", "p2", "p3"]].values, fmt='%s')

    os.remove(face_file)
    os.remove(vert_file)
    pass

def merge_cylinders_gmsh(min_l, max_l, r,st, h,dist, mesh_alg=1, mesh_2nd_order=1,\
                         mesh_file_ver=1, save_stl=True, save_msh=True,\
                         center=[0, 0,0],gmsh = '/Applications/Gmsh.app/Contents/MacOS/gmsh ',\
                         mesh_out_name='merged'):
        
    mesh_info = """SetFactory(\"OpenCASCADE\");
Mesh.CharacteristicLengthMin=%3.1f;
Mesh.CharacteristicLengthMax=%3.1f;
Mesh.Algorithm=%d;
Mesh.SecondOrderLinear=%d;
Mesh.MshFileVersion=%d;
r = %7.5f;
st = %7.5f;
dist = %7.5f;
fillet_radius=r/2.0;
h = %7.5f;
x_c =  %7.5f;
y_c =  %7.5f;
z_c =  %7.5f;
Cylinder(1) = {x_c, y_c-h/2, z_c, 0, h, 0, r+st, 2*Pi};
Cylinder(2) = {x_c+2*r+dist, y_c-h/2, z_c, 0, h, 0, r+st, 2*Pi};
BooleanUnion{ Volume{2}; Delete; }{ Volume{1}; Delete; }
Fillet{1}{3, 6}{fillet_radius}
Mesh.RemeshParametrization = 1;
Mesh.Smoothing = 1;
Mesh 2;""" % (min_l, max_l, mesh_alg, mesh_2nd_order, mesh_file_ver, r,st, dist, h,center[0],center[1],center[2])
    if save_msh:
        mesh_info += '\n'+"""Save StrCat(StrPrefix(General.FileName),\".msh\");"""
        
    if save_stl:
        mesh_info += '\n'+"""Save StrCat(StrPrefix(General.FileName),\".stl\");""" 
    
    mesh_geo_file = "%s.geo"%mesh_out_name
    write_file_info(mesh_geo_file, mesh_info)
    mesh_gen_cmd = "%s%s%s" % (gmsh, mesh_geo_file ,' -0')
    os.system(mesh_gen_cmd)
    generate_vert_face_from_msh(mesh_out_name,'merged_Stern')
    pass

def ion_pmf(min_l, max_l, r1, r2, st=2, dist=0, mesh_alg=1, mesh_2nd_order=1,\
                         mesh_file_ver=1, save_stl=True, save_msh=True,\
                         center=[0, 0,0],gmsh = '/Applications/Gmsh.app/Contents/MacOS/gmsh ',\
                         mesh_name='merged_Stern'):
        
    mesh_info = """SetFactory(\"OpenCASCADE\");
Mesh.CharacteristicLengthMin=%3.1f;
Mesh.CharacteristicLengthMax=%3.1f;
Mesh.Algorithm=%d;
Mesh.SecondOrderLinear=%d;
Mesh.MshFileVersion=%d;
r1 = %3.1f;
r2 = %3.1f;
st = %3.1f;
d = %3.1f;
x_c =  %4.2f;
y_c =  %4.2f;
z_c =  %4.2f;
sp1=newreg;
Sphere(sp1) = {x_c, y_c, z_c, r1+st,-Pi/2, Pi/2, 2*Pi};
If (r2!=0)
sp2=newreg;
Sphere(sp2) = {x_c+r2+d+r1, y_c, z_c, r2+st,-Pi/2, Pi/2, 2*Pi};
BooleanUnion{ Volume{sp2}; Delete; }{ Volume{sp1}; Delete; }
fillet_radius=(r1+r2+d)/2.0;
Fillet{1}{5, 4}{fillet_radius}
EndIf
Mesh 2;
RefineMesh;""" % (min_l, max_l, mesh_alg, mesh_2nd_order, mesh_file_ver, r1, r2, st, dist, center[0],center[1],center[2])
    if save_msh:
        mesh_info += '\n'+"""Save StrCat(StrPrefix(General.FileName),\".msh\");"""
        
    if save_stl:
        mesh_info += '\n'+"""Save StrCat(StrPrefix(General.FileName),\".stl\");""" 
    
    mesh_geo_file = "%s.geo"%mesh_name
    write_file_info(mesh_geo_file, mesh_info)
    mesh_gen_cmd = "%s%s%s" % (gmsh, mesh_geo_file ,' -0')
    os.system(mesh_gen_cmd)
    generate_vert_face_from_msh(mesh_name, mesh_name)
    pass

def translate_df_to(df, point):
    
    df.loc[:,["X"]] += point[0]
    df.loc[:,["Y"]] += point[1]
    df.loc[:,["Z"]] += point[2]
    return df

def generate_ion_pmf_problem(r_left, r_right, distance, stern, min_l, max_l, left_charge=-1, right_charge=1,\
                             right_mesh_name='molDiel',left_mesh_name='memDiel',merged_name='merged_Stern',\
                             stern_mesh=False, joung_radii=False):
    
    joung_radii_dict_scaled = {-1:2.32, 1:1.26}
    atom_name_dict = {1:'Na',-1:'Cl'}
    
    if joung_radii:
        r_left=joung_radii_dict_scaled.get(left_charge,'radius not found!')
        r_right=joung_radii_dict_scaled.get(right_charge,'radius not found!')
    
    right_center = [0+distance+r_left+r_right, 0,0]
    left_center = [0, 0,0]
    
    if distance<2*stern:
        
        ion_pmf(min_l, max_l, r_left, r_right, st=stern, dist=distance, mesh_name=merged_name) #merged
    
    if stern_mesh:
        right_mesh_name='molStern'
        left_mesh_name='memStern'
        r_left += stern
        r_right += stern
        min_l *= 2.0
        max_l *= 2.0
    left_right_dict = {'left':['membrane',left_mesh_name, left_charge, left_center, r_left],\
                       'right':['mol',right_mesh_name, right_charge, right_center, r_right]}
    for direction_ in ['left','right']:
        charge_ = left_right_dict.get(direction_,"Not found")[2]
        center_ = left_right_dict.get(direction_,"Not found")[3]
        mesh_name_ = left_right_dict.get(direction_,"Not found")[1]
        r_ = left_right_dict.get(direction_,"Not found")[4]
        ion_pmf(min_l/2.0, max_l/2.0, r_, 0,0, center=center_, mesh_name=mesh_name_) 
        f = open('%s.pqr' % left_right_dict.get(direction_,"Not found")[0],"w+")
        f.write("%s %5d %4s %5s %5d %9.6f %9.6f %9.6f %9.6f %9.6f\n" % \
        ('ATOM',1, atom_name_dict.get(charge_,'Atom not found'),\
        'TMP',1, center_[0],center_[1],center_[2],charge_, 1))
        f.close()
    pass

class ProblemParams(object):
    
    def __init__(self, hex_pack_dist=9.18, kappa=1e-12, stern_layer_t=2,\
                 charge_to_side_tol=10, charge_to_surf_tol=0, epsilon_w=78.5,\
                 epsilon_p=1, alpha_asym=0.9, beta_asym=-20, gamma_asym=0.01, mu_asym=0.4,\
                 pqr_file='smallScale.pqr',problem_name='Membrane',top_only=False):
        
        self.hex_pack_dist=hex_pack_dist
        self.kappa=kappa
        self.stern_layer_t=stern_layer_t
        self.charge_to_side_tol=charge_to_side_tol
        self.charge_to_surf_tol=charge_to_surf_tol
        self.epsilon_w=epsilon_w
        self.epsilon_p=epsilon_p
        self.alpha_asym=alpha_asym
        self.alpha_asym=beta_asym
        self.alpha_asym=gamma_asym
        self.alpha_asym=mu_asym
        self.pqr_file=pqr_file
        self.problem_name=problem_name
        self.top_only=top_only
        
    def get_hex_pack_dist(self):
        return self.hex_pack_dist
    
    def get_charge_to_side_tol(self):
        return self.charge_to_side_tol
    
    def get_charge_to_surf_tol(self):
        return self.charge_to_surf_tol
    
    def get_stern_layer_t(self):
        return self.stern_layer_t
    
    def get_pqr_file(self):
        return self.pqr_file
    
    pass
    
    

## Charge Distribution class

class ChargeDist(object):
    
    def __init__(self, geometry, params, save_xyzr=True, top_only=False, repo_path=path_to_repo):
        
        self.pqr_file=params.get_pqr_file()
        self.tolerance=params.get_charge_to_surf_tol()
        self.geometry=geometry
        self.top_only=top_only
        """if (self.geometry in ['Sphere','Cylinder']):
            if not self.get_thickness():
                self.top_only = True"""
        self.save_xyzr=save_xyzr
        self.repo_path=repo_path
        
    def generate_pqr(self):
    
        """
        Reads the pqr file of a single monomer chain
        Centers it to the center of the membrane
        Aligns it perpendicularly to the surface of the membrane
        Returns the pqr of the central monomer of the top and bottom surfaces assuming that the center of the 
        bottom surface passes (0, 0,0). 
        tol argument determines the tolerance between the bottom of the bottom atom (z = z_min) of the 
        monomer (closest to the water region) and the bottom dielectric surface of the membrane (distance to (0, 0,0))
        (or the topmost point on the top atom and the top dielectric surface (0, 0,t))
        The input monomer chain is assumed to be perpendicular to x-y plane with headgroups on top and hydrophobic 
        groups on the bottom of the chain
        """
        import os
        file_name = os.path.join(self.repo_path,('%s' % self.pqr_file))
        t = self.geometry.get_thickness()
        tol = self.tolerance

        with open(file_name) as f:

            pqr = [line.rstrip('\n').split() for line in f]

        f.close()

        # array structure of the pqr file
        pqr_data = np.reshape(pqr, np.shape(pqr))

        # Positions (p)
        init_pos = pqr_data[:,:3].astype(float)

        # Charges
        charge = pqr_data[:,3].astype(float)

        # Radii
        radius = pqr_data[:,4].astype(float)

        # center of the monomer on x-y plane (cx, cy, 0)
        center = np.zeros(3)
        center[0], center[1] = np.mean(pqr_data[:,0].astype(float)),np.mean(pqr_data[:,1].astype(float))

        # Generate a (#Atoms by 3) size matrix with all the rows equal to (cx, cy, 0) and substract 
        # it from the initial position matrix (#Atoms by 3 also) to translate the monomer chain
        # to the center (0, 0,some z)
        new_pos = (init_pos - np.broadcast_to(center,(pqr_data.shape[0],3))).round(6)

        # Finding the orientation of the monomer by finding the headgroup locations in pqr_data
        # For now I assume that the monomer structure is given in a configuration that is perpendicular to 
        # the x-y plane (flat-membrane)

        # Finding the top atom (max z)
        top_index = np.argmax(new_pos[:,2])

        # The topmost point on the top atom needs to be tolerance away from the dielectric mesh 
        # of the top surface (at z=0)
        # The current topmost point on the top atom is z_atom + radius_atom
        z_translation = np.float(t - (new_pos[top_index][2] + radius[top_index] + tol))

        final_pos_top = (new_pos + np.broadcast_to([0, 0, z_translation],(pqr_data.shape[0],3))).round(6)
        final_pos_bot = (np.broadcast_to([0, 0, t],(pqr_data.shape[0],3)) - final_pos_top).round(6)

        return final_pos_top, final_pos_bot, charge, radius 
    
    
    def transfer_to_surf(self, params):

        """
        Transfer the initial pqr to (perpendicular to the x-y plane) to top and bottom interfaces of a 
        shell geometry.
        """    
        top, __ = self.geometry.get_charge_positions_normals(params.get_charge_to_side_tol(),params.get_hex_pack_dist())
        thickness=self.geometry.get_thickness()
        pos_top = top.values[:,:3]
        normals_top = -1*top.values[:,3:]
        rot_top = np.zeros((normals_top.shape[0],3, 3))
        nHat_top = np.array([0, 0,-1])
        
        for i in range(normals_top.shape[0]):

            rot_top[i] = get_rot(nHat_top, normals_top[i])

        single_pqr=self.generate_pqr()
        rel_init = single_pqr[0]- np.broadcast_to(np.array([0, 0,thickness]),(single_pqr[0].shape[0],3))
        p_top = np.zeros((pos_top.shape[0],single_pqr[0].shape[0],single_pqr[0].shape[1]))
        p_top = np.tile(pos_top, single_pqr[0].shape[0]).reshape(pos_top.shape[0],single_pqr[0].shape[0],3)+np.matmul(rot_top, rel_init.T).transpose(0, 2,1)
        p_top = p_top.reshape((pos_top.shape[0]*single_pqr[0].shape[0]),3)

        q_top = np.round(np.tile(single_pqr[2], (1, pos_top.shape[0])).T, 6)
        r_top   = np.round(np.tile(single_pqr[3], (1, pos_top.shape[0])).T, 6)
        pqr_top =  np.concatenate([p_top, q_top, r_top], axis=1)

        if self.top_only:

            pqr_final = pqr_top
            
        else:
            __, bot = self.geometry.get_charge_positions_normals(params.get_charge_to_side_tol(),params.get_hex_pack_dist())
        
            pos_bot = bot.values[:,:3]
            normals_bot = bot.values[:,3:]

            rot_bot = np.zeros((normals_bot.shape[0],3, 3))
            nHat_bot = np.array([0, 0,1])
            
            for j in range(normals_bot.shape[0]):

                rot_bot[j] = get_rot(nHat_bot, normals_bot[j])
            p_bot = np.zeros((pos_bot.shape[1],single_pqr[1].shape[0],single_pqr[1].shape[1]))
            p_bot = np.tile(pos_bot, single_pqr[1].shape[0]).reshape(pos_bot.shape[0],single_pqr[1].shape[0],3)\
                   +np.matmul(rot_bot, single_pqr[1].T).transpose(0, 2,1)
            p_bot = p_bot.reshape((pos_bot.shape[0]*single_pqr[1].shape[0]),3)
            q_bot = np.round(np.tile(single_pqr[2], (1, pos_bot.shape[0])).T, 6)
            r_bot   = np.round(np.tile(single_pqr[3], (1, pos_bot.shape[0])).T, 6)
            pqr_bot =  np.concatenate([p_bot, q_bot, r_bot], axis=1)
            pqr_final=np.concatenate([pqr_top, pqr_bot], axis=0)
        
        if self.save_xyzr:

            xyzr = np.delete(pqr_final, 3,1)
            np.savetxt(('membrane.xyzr'), xyzr, fmt='%5.5f')
            xyzr[:,3] += 2
            np.savetxt(('membrane.xyzr.stern'), xyzr, fmt='%5.5f')
        
        if not self.top_only:

            return pqr_final, pqr_top, pqr_bot
        return pqr_final, pqr_top, 1
    


## Surface Mesh Class

class SurfaceMesh(object):
    
    def __init__(self, min_char_length, max_char_length, params, algorithm=6,
                 second_order_linear=1, file_version=1, refine_in_gmsh=False,
                 gmsh_folder='/Applications/Gmsh.app/Contents/MacOS/gmsh ',
                 stl_file=True, msh_file=True, refine_in_gmsh_order=0):
        
        self.min_char_length=min_char_length
        self.max_char_length=max_char_length
        self.algorithm=algorithm
        self.order=second_order_linear
        self.file_version=file_version
        self.gmsh_folder=gmsh_folder
        self.stl_file=stl_file
        self.msh_file=msh_file
        self.charge_to_side_tol=params.get_charge_to_side_tol()
        self.stern_layer_t=params.get_stern_layer_t()
        self.hex_pack_dist=params.get_hex_pack_dist()
        self.msh_file=msh_file
        self.refine_in_gmsh=refine_in_gmsh
        self.refine_in_gmsh_order = refine_in_gmsh_order
        
    def get_info(self):
        
        if self.refine_in_gmsh:
            return self.min_char_length*(2**(self.refine_in_gmsh_order)),\
                   self.max_char_length*(2**(self.refine_in_gmsh_order)),\
                   self.algorithm, self.order, self.file_version
    
        return self.min_char_length, self.max_char_length, self.algorithm, self.order, self.file_version
    
    def get_gmsh_info(self):
        
        mesh_info = """SetFactory(\"OpenCASCADE\");
Mesh.CharacteristicLengthMin=%3.1f;
Mesh.CharacteristicLengthMax=%3.1f;
Mesh.Algorithm=%3.1f;
Mesh.SecondOrderLinear=%d;
Mesh.MshFileVersion=%d;""" % (self.get_info())
        
        return mesh_info
    
    def get_mesh_save_files(self):
        
        out = """Mesh 2;"""
        if self.refine_in_gmsh:
            for i in range(self.refine_in_gmsh_order):
                out += '\n'+"""RefineMesh;"""
        if self.msh_file:
            out += '\n'+"""Save StrCat(StrPrefix(General.FileName),\".msh\");"""
            
        if self.stl_file:
            out += '\n'+"""Save StrCat(StrPrefix(General.FileName),\".stl\");"""
            
        return out

    
    def generate_mesh(self, geometry):

        """
        Generates mesh files with msh format "msh" cylindrical shell geometries using Gmsh for dielectric 
        and Stern interfaces.
        """
        st = self.stern_layer_t
        if geometry.name == 'Cylinder':
            geom_info_diel = geometry.write_gmsh_info(0, self.charge_to_side_tol, self.hex_pack_dist)
            geom_info_stern = geometry.write_gmsh_info(st, self.charge_to_side_tol, self.hex_pack_dist)
        elif geometry.name == 'Cube':
            geom_info_diel = geometry.write_gmsh_info(0, self.charge_to_side_tol, self.hex_pack_dist)
            geom_info_stern = geometry.write_gmsh_info(st, self.charge_to_side_tol, self.hex_pack_dist)
        elif geometry.name == 'Sphere':
            geom_info_diel = geometry.write_gmsh_info(0, self.charge_to_side_tol, self.hex_pack_dist)
            geom_info_stern = geometry.write_gmsh_info(st, self.charge_to_side_tol, self.hex_pack_dist)
        mesh_info_diel = self.get_gmsh_info()
        self.min_char_length *= 2
        self.max_char_length *= 2
        mesh_info_stern = self.get_gmsh_info()
        gmsh_info_diel = mesh_info_diel+'\n'+geom_info_diel+'\n'+self.get_mesh_save_files()
        gmsh_info_stern = mesh_info_stern+'\n'+geom_info_stern+'\n'+self.get_mesh_save_files()
        # Writing diel mesh
        geo_file_name_diel = "%s_diel.geo" % (geometry.geo_file_name)
        geo_file_name_stern = "%s_stern.geo" % (geometry.geo_file_name)
        msh_file_name_diel = "%s_diel.msh" % (geometry.geo_file_name)
        msh_file_name_stern = "%s_stern.msh" % (geometry.geo_file_name)
        #diel_mesh = pymesh.load_mesh(msh_file_name_diel)
        #stern_mesh = pymesh.load_mesh(msh_file_name_stern)
        write_file_info(geo_file_name_diel, gmsh_info_diel)
        write_file_info(geo_file_name_stern, gmsh_info_stern)
        
        # Converting *.geo files to *.stl using Gmsh
        gmsh = self.gmsh_folder #MacOS
        mesh_gen_cmd_diel = "%s%s%s" % (gmsh, geo_file_name_diel ,' -0')
        mesh_gen_cmd_stern= "%s%s%s" % (gmsh, geo_file_name_stern,' -0')
        os.system(mesh_gen_cmd_diel)
        os.system(mesh_gen_cmd_stern)
        pass

        
    
def get_msh_file_name(self, geometry):
        
        msh_file_name_diel = "%s_%s" % (geometry.geo_file_name,'diel')
        msh_file_name_stern = "%s_%s" % (geometry.geo_file_name,'stern')
        
        return [msh_file_name_diel, msh_file_name_stern]


## Problem Class

class ElectrostaticProblem(object):
    
    stern=True
    
    def __init__(self, params, geometry, charges, mesh, name='membrane'):
        
        self.params=params
        self.geometry=geometry
        self.charges=charges
        self.mesh=mesh
        self.name=name
        
    def gen_vf(self):
    
        """
        generates vertices and faces of a mesh from a "msh" file using a batch script.
        """

        pwd = os.getcwd()
        cp_cmd = 'cp /Users/Ali/repos/ali_SLIC-Membrane/generate_vert_face.sh %s' % pwd

        subprocess.Popen(cp_cmd, shell=True).wait()
        chmod_cmd = 'chmod +x generate_vert_face.sh'
        subprocess.Popen(chmod_cmd, shell=True).wait()
        
        for interface in ['diel','stern']:
            
            mesh_name = "%s_%s" % (self.geometry.geo_file_name, interface)
            mesh_out_name = "%s_%s" % (self.name, interface)
            generate_vert_face_from_msh(mesh_name, mesh_out_name)

        pass

    def pqr_to_PyGBe(self, return_df=False):
        
        pqr, __, __ = self.charges.transfer_to_surf(self.params)
        single_pqr = self.charges.generate_pqr()
        file_name = self.charges.pqr_file
        
        atom_type = ['ATOM'] * int(pqr.shape[0])
        df1=pd.DataFrame(atom_type, columns=["Type"])
        atom_num = np.arange(1, pqr.shape[0]+1)
        df2=pd.DataFrame(atom_num, columns=["Atom_number"])
        atom = ['A'] * int(pqr.shape[0])
        df3=pd.DataFrame(atom, columns=["Atom_type"])
        aa_type = ['AA'] * int(pqr.shape[0])
        df4=pd.DataFrame(aa_type, columns=["AA_type"])
        single_len = single_pqr[0].shape[0]
        chain = np.zeros(pqr.shape[0])

        for i in range(int(pqr.shape[0]/single_len)):

            for j in range(single_len):

                chain[j+i*single_len]=i+1

        df5=pd.DataFrame(chain, columns=["Chain"])
        df=pd.DataFrame(pqr, columns=["X", "Y", "Z", "q", "r"])
        df = pd.concat([df1, df2, df3, df4, df5, df], axis=1)
        df[["Atom_number", "Chain"]] = df[["Atom_number", "Chain"]].apply(pd.to_numeric)
        df[["X", "Y", "Z", "q", "r"]] = df[["X", "Y", "Z", "q", "r"]].apply(pd.to_numeric)
        df['Type'] = df['Type'].map('{0:5s}'.format)
        df['Atom_number'] = df['Atom_number'].map('{0:6d}'.format)
        df['Atom_type'] = df['Atom_type'].map('{0:6s}'.format)
        df['AA_type'] = df['AA_type'].map('{0:5s}'.format)
        df['Chain'] = df['Chain'].map('{0:.0f}'.format)
        df['X'] = df['X'].map('{0:14.6f}'.format)
        df['Y'] = df['Y'].map('{0:12.6f}'.format)
        df['Z'] = df['Z'].map('{0:12.6f}'.format)
        df['q'] = df['q'].map('{0:11.6f}'.format)
        df['r'] = df['r'].map('{0:11.6f}'.format)
        np.savetxt("%s.pqr" % self.name, df.values, fmt='%s')

        if return_df:
            return df
        pass
        
    
    def visualize(self, azim = -90, elev = 0, show_stern = False, transparency = 0.1,\
              make_movie = False, show = True, pqr_point_area = 0.1):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        """
        Visualization of the problem
        """
        pqr, __, __=self.charges.transfer_to_surf(self.params)
        pwd =os.getcwd()
        diel_file = "%s_%s" % (self.name,'diel')
        stern_file = "%s_%s" % (self.name,'stern')
        p_x = pqr[:,0]
        p_y = pqr[:,1]
        p_z = pqr[:,2]
        c = np.arange(1, 1000)

        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
        v_x_diel, v_y_diel, v_z_diel, face_diel = trisurf_from_mesh(diel_file)
        diel_file_in= '%s_in'%diel_file

        path = os.path.join(pwd,"%s.vert"%diel_file_in)

        if os.path.isfile(path):

            v_x_diel_in, v_y_diel_in, v_z_diel_in, face_diel_in = trisurf_from_mesh(diel_file_in)

        if show_stern:

            v_x_stern, v_y_stern, v_z_stern, face_stern = trisurf_from_mesh(stern_file)
            if os.path.isfile(path):

                stern_file_in= '%s_in'%stern_file
                v_x_stern, v_y_stern, v_z_stern, face_stern_in = trisurf_from_mesh(stern_file_in)

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot_trisurf(v_x_diel, v_y_diel, v_z_diel, triangles=face_diel, linewidth=0.6, facecolors ='r', edgecolors=cmap.to_rgba(int(transparency*1000)),alpha = transparency, antialiased=True)
        #ax.plot_trisurf(x, y, z, linewidth=2, antialiased=True, edgecolors=c, facecolors=c) #modified this line

        ax1.scatter(p_x, p_y, p_z, marker='o', s=pqr_point_area, c="goldenrod")

        if os.path.isfile(path):

            ax1.plot_trisurf(v_x_diel_in, v_y_diel_in, v_z_diel_in, triangles=face_diel_in, color ='y', edgecolors='g',linewidth=0.1, alpha = transparency, antialiased=True)

            if show_stern:

                ax1.plot_trisurf(v_x_stern_in, v_y_stern_in, v_z_stern_in, triangles=face_stern_in, color ='m', edgecolors='b',linewidth=0.1, alpha = transparency, antialiased=True)

        if show_stern:

            ax1.plot_trisurf(v_x_stern, v_y_stern, v_z_stern, triangles=face_stern, color ='y', edgecolors='r',linewidth=0.1, alpha = transparency, antialiased=True)

        ax1.set_axis_off()
        ax1.view_init(azim, elev)
        scaling = np.array([getattr(ax1, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        ax1.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)

        if show:

            return plt

        else:

            plt.savefig('pqr.png', dpi=300, facecolor='w', edgecolor='w')
            plt.close(fig)

            pass

## Sphere Geometry Class

class Sphere(object):
    
    #stern_layer_t = 2; Example: https://www.python-course.eu/python3_class_and_instance_attributes.php
    """
    Blueprint for Geometry
    """
    
    def __init__(self, r1, r2, theta=np.pi, center=[0, 0,0],name='Sphere'):
        
        self.name = name
        self.theta = theta
        assert (r1 != r2), "%s=%s: The two radii can NOT be equal in a spherical shell"\
        %(r1, r2)
        assert ((r1 > 0) and (r2 >= 0)), "r_1 and %s must be greater than or equal to zero"\
        %(r1, r2)
        self.r1 = max(r1, r2)
        self.r2 = min(r1, r2)
        if (self.r2 == 0):
            self.thickness = 0
            self.theta = np.pi
        else:
            self.thickness = self.r1-self.r2
        self.geo_file_name = "sphere_%s_%s_%s" % (self.r1, self.r2, np.round(self.theta, 1))
        self.theta_mesh = theta
        self.theta_charged = [theta, theta]
        self.center=center
        
    def __repr__(self):
        
        return 'Sphere(r1={},r2={})'.format(self.r1, self.r2)
    
    def __str__(self):
        
        if not (self.r2 == 0):
            return 'This is a spherical shell with outer radius of {:.2f}, inner radius of {:.2f}'.format(self.r1, self.r2)
        return 'This is a sphere radius of {:.2f}'.format(self.r1)
 
    def get_area(self):
        """Return the surface area of this Sphere instance"""
        
        return 2*(1-math.cos(self.theta_mesh))*np.pi*(self.r1**2+self.r2**2)
    
    def get_mesh_area(self):
        return self.get_area()
    
    def get_charged_area(self):
        return 2*np.pi*((1-math.cos(self.theta_charged[0]))*self.r1**2+(1-math.cos(self.theta_charged[1]))*self.r2**2)
    
    def get_volume(self):
        
        return 4/3*np.pi*((self.r1)**3-(self.r2)**3)
    
    def get_avg_radius(self):
        
        if self.get_thickness():
            return (self.r1+self.r2)/2.0
        return self.r1
    
    def get_name(self):
        
        return self.name
    
    def get_thickness(self):
        
        return self.thickness
    
    def get_geo_file_name(self):
        
        return self.geo_file_name
    
    def write_gmsh_info(self, st, charge_to_side_tol, hex_pack_dist):
        
        if not self.get_thickness():
            info = """r_out = %7.5f;
Sphere(1) = {0, 0, 0, r_out,-Pi/2, Pi/2, 2*Pi};""" % (self.r1+st)
        else:
            if self.theta==np.pi:
                info = """r_out = %7.5f;
r_in = %7.5f;
Sphere(1) = {0, 0, 0, r_out,-Pi/2, Pi/2, 2*Pi};
Sphere(2) = {0, 0, 0, r_in,-Pi/2, Pi/2, 2*Pi};""" % (self.r1+st, self.r2-st)
            else:
                info = """r_out = %7.5f;
r_in = %7.5f;
theta = %7.5f;
x_c = %7.5f;
y_c = %7.5f;
z_c = %7.5f;
Sphere(1) = {x_c, y_c, z_c, r_out, Pi/2-theta, Pi/2, 2*Pi};
Sphere(2) = {x_c, y_c, z_c, r_in,-Pi/2, Pi/2, 2*Pi};
Cone(3) = {x_c, y_c, z_c, 0, 0, r_out*Cos(theta), 0, r_out*Sin(theta), 2*Pi};
If (r_out*Cos(theta) < 0)
BooleanDifference{ Volume{1}; Delete; }{ Volume{3}; Delete; }
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
Else
BooleanUnion{ Volume{3}; Delete; }{ Volume{1}; Delete; }
BooleanDifference{ Volume{3}; Delete; }{ Volume{2}; Delete; }
EndIf""" % (self.r1+st, self.r2-st, self.theta_mesh+st/(self.get_avg_radius()),\
           self.center[0],self.center[1],self.center[2])
        
        return info
    
    def get_charge_positions_normals(self, charge_to_side_tol, hex_pack_dist):
    
        density = 2/hex_pack_dist**2/np.sqrt(3)
        
        num_pts_top = int(4*np.pi*self.r1**2*density)
        num_pts_bot = int(4*np.pi*self.r2**2*density)

        indices_top = np.arange(0, num_pts_top, dtype=float) + 0.5
        indices_bot = np.arange(0, num_pts_bot, dtype=float) + 0.5

        phi_top = np.arccos(1 - 2*indices_top/num_pts_top)
        theta_top = np.pi * (1 + 5**0.5) * indices_top

        phi_bot = np.arccos(1 - 2*indices_bot/num_pts_bot)
        theta_bot = np.pi * (1 + 5**0.5) * indices_bot

        n_x_top, n_y_top, n_z_top = np.cos(theta_top) * np.sin(phi_top), np.sin(theta_top) * np.sin(phi_top),np.cos(phi_top)

        top = pd.DataFrame({'X' :(self.r1)*n_x_top,'Y' :(self.r1)*n_y_top,'Z' :(self.r1)*n_z_top,\
                            'nX':n_x_top,'nY':n_y_top,'nZ':n_z_top})
        top=top[(top.Z>self.r1*np.cos(self.theta))]
        n_x_bot, n_y_bot, n_z_bot = np.cos(theta_bot) * np.sin(phi_bot), np.sin(theta_bot) * np.sin(phi_bot),np.cos(phi_bot)
        bot = pd.DataFrame({'X' :(self.r2)*n_x_bot,'Y' :(self.r2)*n_y_bot,'Z' :(self.r2)*n_z_bot,\
                            'nX':n_x_bot,'nY':n_y_bot,'nZ':n_z_bot})
        bot=bot[(bot.Z>self.r2*np.cos(self.theta))]
        max_theta_id_top = top.Z.idxmin()
        max_theta_id_bot = bot.Z.idxmin()
        max_theta_top = np.arccos(abs(top.iloc[max_theta_id_top].Z)/self.r1)
        max_theta_bot = np.arccos(abs(bot.iloc[max_theta_id_bot].Z)/self.r2)
        if abs(self.theta - 3*np.pi/4) < np.pi/4:
            max_theta_top += np.pi-max_theta_top
            max_theta_bot += np.pi-max_theta_bot
        max_theta = max(max_theta_bot, max_theta_top)
        top = translate_df_to(top, self.center)
        bot = translate_df_to(bot, self.center)
        setattr(self,"theta_mesh",max_theta+charge_to_side_tol/self.get_avg_radius())
        setattr(self,"theta_charged",[max_theta_bot+ hex_pack_dist*np.sqrt(3)/4/self.r2, max_theta_top+hex_pack_dist*np.sqrt(3)/4/self.r1])
        return top, bot


## Cylinder Geometry Class

class Cylinder(object):
    
    #stern_layer_t = 2; Example: https://www.python-course.eu/python3_class_and_instance_attributes.php
    """
    Blueprint for Geometry
    """
    
    def __init__(self, r1, r2, h, theta, name='Cylinder',center=[0, 0,0],\
                 thickness_for_full_cyl=10.125, fill_if_full=True, side=True):
        
        assert (r1 != r2), "%s=%s: The two radii can NOT be equal in a cylindrical shell"\
        %(r1, r2)
        assert ((r1 >= 0) and (r2 >= 0)), "%s and %s must be greater than or equal to zero"\
        %(r1, r2)
        self.r1 = max(r1, r2)
        self.r2 = min(r1, r2)
        self.h = h 
        self.name = name
        self.theta = theta
        self.geo_file_name = "cyl_%s_%s" % (self.r1, self.r2)
        self.side=side
        self.center=center
        
        if (self.r2 != 0):
            assert self.side, "Wrong setting!"
            self.thickness = self.r1-self.r2
        else:
            assert (self.theta == 2*np.pi), "Tube has to be closed!"
            self.thickness = 0
        self.theta_mesh = theta
        self.theta_charged = [theta, theta]
        self.h_mesh = h
        self.h_charged = h
        
    def __repr__(self):
        """Return a formal string that can be used to re-create this instance, invoked by repr()"""
        
        return 'Cylinder(r1={},r2={},h={},theta={})'.format(self.r1, self.r2, self.h, self.theta)
    
    def __str__(self):
        """Return a descriptive string for this instance, invoked by print() and str()"""
        
        return 'This is a cylindrical shell with outer radius of {:.2f}, inner radius of {:.2f},\
height of {:.2f} and angle {:.2f} radians ({:.2f} degrees)'.format(self.r1, self.r2, self.h, self.theta, self.theta*180/np.pi)
 
    def get_area(self):
        """Return the surface area of this Cylinder instance"""
        
        return 2*self.theta*(self.r1**2-self.r2**2+(self.r1+self.r2)*self.h)
    
    def get_mesh_area(self):
        """Return the surface area of this Cylinder instance"""
        
        return 2*self.theta_mesh*self.get_avg_radius()*self.h_mesh
    
    def get_charged_area(self):
        """Return the surface area of this Cylinder instance"""
        
        return (self.theta_charged[0]*self.r1+self.theta_charged[1]*self.r2)*self.h_charged
    
    def get_volume(self):
        
        return self.theta*((self.r1)**2-(self.r2)**2)*(self.h_mesh)
    
    def get_name(self):
        
        return self.name
    
    def get_thickness(self):
        
        return self.thickness
    
    def get_geo_file_name(self):
        
        return self.geo_file_name
    
    def get_avg_radius(self):
        
        if self.get_thickness():
            return (self.r1+self.r2)/2.0
        return self.r1
    
    def get_charge_positions_normals(self, charge_to_side_tol, hex_pack_dist):

        """
        Returns new charge locations on a cylindrical shell and normal to the dielectric surface at 
        that points which is used to trnsform the initial pqr.
        """    

        
        n_half_seg_hor, n_seg_ver = get_flat_membrane_info(self.get_avg_radius()*self.theta, self.h, charge_to_side_tol, hex_pack_dist)

        if self.theta != 2*np.pi:
            
            theta_top = n_half_seg_hor*hex_pack_dist*np.sqrt(3)/2/self.get_avg_radius()
        else:
            theta_top = 2*np.pi
            
        theta_bot = theta_top    
        h = n_seg_ver*hex_pack_dist
        setattr(self,"h_charged",h+hex_pack_dist)
        setattr(self,"h_mesh",h+2*charge_to_side_tol)
        if self.side:
            
            n_half_seg_hor_top = int((self.r1)*theta_top/(hex_pack_dist*np.sqrt(3)/2))
            n_half_seg_hor_top = odd_it(n_half_seg_hor_top)
            
            n_points_hor_top = int(n_half_seg_hor_top/2) + 1
            n_points_ver = n_seg_ver + 1

            theta_top = n_half_seg_hor_top*hex_pack_dist*np.sqrt(3)/2/(self.r1)
            
            """
               ^<------------------------L_charged-------------------------->^
               |    *     *     *     *     *     *     *     *     *     * | 
               | @     @     @     @     @     @     @     @     @     @    |
               |    *     *     *     *     *     *     *     *     *     * | 
               | @     @     @     @     @     @     @     @     @     @    |                                  ____ 
               |    *     *     *     *     *     *     *     *     *     * |                                 |    |
              H| @     @     @     @     @     @     @     @     @     @    |H_charged - - - ->  charged area=|    |
               |    *     *     *     *     *     *     *     *     *     * |                                 |____|
               | @     @     @     @     @     @     @     @     @     @    |
               |    *     *     *     *     *     *     *     *     *     * |
     inner ==> | @     @     @     @     @     @     @     @     @     @    | 
     outer ==> |    *     *     *     *     *     *     *     *     *     * |
               ^<----------------------------L----------------------------->^
            """

            d_theta_top = hex_pack_dist*np.sqrt(3)/2/(self.r1)
            theta_top_outer = np.linspace(-theta_top/2+d_theta_top, theta_top/2, n_points_hor_top)
            theta_top_inner = np.linspace(-theta_top/2, theta_top/2-d_theta_top, n_points_hor_top)
            h_outer = np.linspace(-h/2, h/2, n_points_ver)
            h_inner = np.linspace(-h/2+hex_pack_dist/2, h/2-hex_pack_dist/2, n_points_ver-1)

            if self.r2 != 0:
                
                n_half_seg_hor_bot = int((self.r2)*theta_bot/(hex_pack_dist*np.sqrt(3)/2))
                n_half_seg_hor_bot = odd_it(n_half_seg_hor_bot)
                n_points_hor_bot = int(n_half_seg_hor_bot/2) + 1

                theta_bot = n_half_seg_hor_bot*hex_pack_dist*np.sqrt(3)/2/(self.r2)
                d_theta_bot = hex_pack_dist*np.sqrt(3)/2/(self.r2)
                theta_bot_outer = np.linspace(-theta_bot/2+d_theta_bot, theta_bot/2, n_points_hor_bot)
                theta_bot_inner = np.linspace(-theta_bot/2, theta_bot/2-d_theta_bot, n_points_hor_bot)
                
                if (theta_top + 2*charge_to_side_tol/(self.r2)) > 2*np.pi and self.fill_if_full:

                    n_points_hor_top = int(2*np.pi*self.r1/hex_pack_dist/np.sqrt(3))
                    tet_top = (2*np.pi*self.r1-(n_points_hor_top-1)*hex_pack_dist*np.sqrt(3))/(self.r1)/2
                    theta_top_outer = np.linspace(tet_top, 2*np.pi-tet_top-d_theta_top, n_points_hor_top)
                    theta_top_inner = np.linspace(tet_top+d_theta_top, 2*np.pi-tet_top, n_points_hor_top)

                    n_points_hor_bot = int(2*np.pi*self.r2/hex_pack_dist/np.sqrt(3))
                    tet_bot = (2*np.pi*self.r2-(n_points_hor_bot-1)*hex_pack_dist*np.sqrt(3))/(self.r2)/2
                    theta_bot_outer = np.linspace(tet_bot, 2*np.pi-tet_bot-d_theta_bot, n_points_hor_bot)
                    theta_bot_inner = np.linspace(tet_bot+d_theta_top, 2*np.pi-tet_bot, n_points_hor_bot)
                
                if n_points_hor_bot == 1:

                    theta_bot_outer = [-d_theta_bot]
                    theta_bot_inner = [d_theta_bot]
                    
                points_bot = []
                points_bot.append([get_cylinder_positions_and_normals(self.r2, j,i) for i in theta_bot_inner for j in h_inner]) 
                points_bot.append([get_cylinder_positions_and_normals(self.r2, j,i) for i in theta_bot_outer for j in h_outer])
                bot_array = np.concatenate([np.array(i) for i in points_bot])
                bot = pd.DataFrame({'X' :bot_array[:,0],'Y' :bot_array[:,1],'Z' :bot_array[:,2],\
                                    'nX':bot_array[:,3],'nY':bot_array[:,4],'nZ':bot_array[:,5]})
            
            if n_points_hor_top == 1:

                theta_top_outer = [-d_theta_top]
                theta_top_inner = [d_theta_top]

            points_top = []
            points_top.append([get_cylinder_positions_and_normals(self.r1, j,i) for i in theta_top_outer for j in h_outer])
            points_top.append([get_cylinder_positions_and_normals(self.r1, j,i) for i in theta_top_inner for j in h_inner])
            
            top_array = np.concatenate([np.array(i) for i in points_top])
            top = pd.DataFrame({'X' :top_array[:,0],'Y' :top_array[:,1],'Z' :top_array[:,2],\
                                    'nX':top_array[:,3],'nY':top_array[:,4],'nZ':top_array[:,5]})
            
            top = translate_df_to(top, self.center)
            
            if self.r2 != 0:  
                
                if theta_top>theta_bot:

                    theta_final = theta_top + 2*charge_to_side_tol/self.r1
                else:
                    theta_final = theta_bot + 2*charge_to_side_tol/self.r2
                
                theta_charged_top = theta_top + hex_pack_dist*np.sqrt(3)/2/self.r1
                theta_charged_bot = theta_bot + hex_pack_dist*np.sqrt(3)/2/self.r2
                setattr(self,"theta_mesh",theta_final)
                setattr(self,"theta_charged",[theta_charged_top, theta_charged_bot])
                top = translate_df_to(top, self.center)
                bot = translate_df_to(bot, self.center)
                return top, bot
            return top, 1
                
        l_cube=(self.r1+charge_to_side_tol)*3
        tmp = Cube(l_cube, l_cube, self.h)
        top, bot = tmp.get_charge_positions_normals(charge_to_side_tol, hex_pack_dist)
        top = top[(top.X**2+top.Y**2 < self.r1**2)]
        bot = bot[(bot.X**2+bot.Y**2 < self.r1**2)]
        top = translate_df_to(top, self.center)
        bot = translate_df_to(bot, self.center)
        
        return top, bot
    
    def get_points_and_normals_full_cylinder_from_n(self, n,params):

        hex_pack_dist = params.hex_pack_dist
        thickness = self.thickness
        charge_to_side_tol = params.charge_to_side_tol
        k = int(np.ceil(2*np.pi*thickness/hex_pack_dist/np.sqrt(3)))
        r = (n+k)*hex_pack_dist*np.sqrt(3)/2/np.pi-thickness/2
        m = int ((h-2*charge_to_side_tol)/hex_pack_dist)
        h =  m*hex_pack_dist 
        n_points_ver = m+1

        n_points_hor_top = k + n
        n_points_hor_bot = n

        tet_bot = hex_pack_dist*np.sqrt(3)/(r-thickness/2)/2
        tet_top = hex_pack_dist*np.sqrt(3)/(r+thickness/2)/2

        theta_top_outer = np.linspace(0, 2*np.pi-hex_pack_dist*np.sqrt(3)/(r+thickness/2),n_points_hor_top)
        theta_top_inner = np.linspace(hex_pack_dist*np.sqrt(3)/(r+thickness/2)/2, 2*np.pi-hex_pack_dist*np.sqrt(3)/(r+thickness/2)/2, n_points_hor_top)

        theta_bot_outer = np.linspace(0, 2*np.pi-hex_pack_dist*np.sqrt(3)/(r-thickness/2),n_points_hor_bot)
        theta_bot_inner = np.linspace(hex_pack_dist*np.sqrt(3)/(r-thickness/2)/2, 2*np.pi-hex_pack_dist*np.sqrt(3)/(r-thickness/2)/2, n_points_hor_bot)

        h_outer = np.linspace(-h/2, h/2, n_points_ver)
        h_inner = np.linspace(-h/2+hex_pack_dist/2, h/2-hex_pack_dist/2, n_points_ver-1)

        points_top = []
        points_bot = []

        points_top.append([calc_cylinder_positions_and_normals(r+thickness/2, j,i) for i in theta_top_outer for j in h_outer])
        points_bot.append([calc_cylinder_positions_and_normals(r-thickness/2, j,i) for i in theta_bot_outer for j in h_outer])
        points_top.append([calc_cylinder_positions_and_normals(r+thickness/2, j,i) for i in theta_top_inner for j in h_inner])
        points_bot.append([calc_cylinder_positions_and_normals(r-thickness/2, j,i) for i in theta_bot_inner for j in h_inner]) 

        top_array = np.concatenate([np.array(i) for i in points_top])
        bot_array = np.concatenate([np.array(i) for i in points_bot])

        top = pd.DataFrame({'X' :top_array[:,0],'Y' :top_array[:,1],'Z' :top_array[:,2],\
                                'nX':top_array[:,3],'nY':top_array[:,4],'nZ':top_array[:,5]})
        bot = pd.DataFrame({'X' :bot_array[:,0],'Y' :bot_array[:,1],'Z' :bot_array[:,2],\
                                'nX':bot_array[:,3],'nY':bot_array[:,4],'nZ':bot_array[:,5]})
        top = translate_df_to(top, self.center)
        bot = translate_df_to(bot, self.center)
        
        return top, bot, r, h

    def write_gmsh_info(self, st, charge_to_side_tol, hex_pack_dist):
        
        #st = mesh.get_stern_thickness(mesh.interface_type)
        if self.side: 
            if self.r2:
                d_theta_st = (2*st)/self.get_avg_radius()
                info = """r_out = %7.5f;
r_in = %7.5f;
t_d = r_out-r_in;
theta = %7.5f;
h = %7.5f;
x_c = %7.5f;
y_c = %7.5f;
z_c = %7.5f;
If (theta < Pi)
Cylinder(1) = {x_c, y_c -h/2, z_c, 0, h, 0, r_out, Pi/2+theta/2};
Cylinder(2) = {x_c, y_c -h/2, z_c, 0, h, 0, r_out, Pi/2-theta/2};
Cylinder(3) = {x_c, y_c -h/2, z_c, 0, h, 0, r_in,  Pi/2+theta/2};
Cylinder(4) = {x_c, y_c -h/2, z_c, 0, h, 0, r_in,  Pi/2-theta/2};
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
BooleanDifference{ Volume{3}; Delete; }{ Volume{4}; Delete; }
BooleanDifference{ Volume{1}; Delete; }{ Volume{3}; Delete; }
Else
If (theta < 2*Pi)
Cylinder(1) = {x_c, y_c -h/2, z_c, 0, h, 0, r_out, 2*Pi};
Cylinder(2) = {x_c, y_c -h/2, z_c, 0, h, 0, r_out, 5*Pi/2-theta/2};
Cylinder(3) = {x_c, y_c -h/2, z_c, 0, h, 0, r_out, Pi/2+theta/2};
Cylinder(4) = {x_c, y_c -h/2, z_c, 0, h, 0, r_in,  2*Pi};
Cylinder(5) = {x_c, y_c -h/2, z_c, 0, h, 0, r_in,  5*Pi/2-theta/2};
Cylinder(6) = {x_c, y_c -h/2, z_c, 0, h, 0, r_in,  Pi/2+theta/2};
BooleanDifference{ Volume{2}; Delete; }{ Volume{3}; Delete; }
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
BooleanDifference{ Volume{5}; Delete; }{ Volume{6}; Delete; }
BooleanDifference{ Volume{4}; Delete; }{ Volume{5}; Delete; }
BooleanDifference{ Volume{1}; Delete; }{ Volume{4}; Delete; }
Else
Cylinder(1) = {x_c, y_c -h/2, z_c, 0, h, 0, r_out, 2*Pi};
Cylinder(2) = {x_c, y_c -h/2, z_c, 0, h, 0, r_in, 2*Pi};
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
EndIf
EndIf"""% (self.r1+st, self.r2-st, self.theta_mesh+d_theta_st, self.h_mesh+2*st,\
           self.center[0],self.center[1],self.center[2])
            
                return info
            info = """r_out = %7.5f;
h = %7.5f;
x_c = %7.5f;
y_c = %7.5f;
z_c = %7.5f;
Cylinder(1) = {x_c, y_c -h/2, z_c, 0, h, 0, r_out, 2*Pi};"""% (self.r1+st, self.h_mesh+2*st,\
                                                         self.center[0],self.center[1],self.center[2])
            return info
        info = """r_out = %7.5f;
h = %7.5f;
st = %7.5f;
x_c = %7.5f;
y_c = %7.5f;
z_c = %7.5f;
Cylinder(1) = {x_c, y_c,-st, 0, 0, h+2*st, r_out+st, 2*Pi};""" % (self.r1+charge_to_side_tol, self.h, st,\
                                                         self.center[0],self.center[1],self.center[2])
                                                       
        return info


## Cube Geometry Class


class Cube(object):
    
    #stern_layer_t = 2; Example: https://www.python-course.eu/python3_class_and_instance_attributes.php
    """
    Blueprint for Geometry
    """
    
    def __init__(self, l, w, h, center=[0, 0,0],name='Cube'):
        
        assert ((l >= 0) and (w >= 0) and (h >= 0)), "%s and %s and %s must be greater than or equal to zero"\
        %(l, w,h)
        self.l = l
        self.w = w
        self.h = h 
        self.name = name
        self.geo_file_name = "cube_%s_%s_%s" % (self.l, self.w, self.h)
        self.l_mesh = l
        self.w_mesh = w
        self.l_charged = l
        self.w_charged = w
        self.center=center
        
    def __repr__(self):
        """Return a formal string that can be used to re-create this instance, invoked by repr()"""
        
        return 'Cube(l={},w={},h={})'.format(self.l, self.w, self.h)
    
    def __str__(self):
        """Return a descriptive string for this instance, invoked by print() and str()"""
        
        return 'This is a cube length of {:.2f}, width of {:.2f} and height {:.2f}'.format(self.l, self.w, self.h)
 
    def get_area(self):
        """Return the surface area of this Cylinder instance"""
        
        return 2*(self.l*self.h+self.l*self.w+self.w*self.h)
    
    def get_mesh_area(self):
        """Return the surface area of this Cylinder instance"""
        
        return 2*(self.l_mesh*self.w_mesh)
    
    def get_charged_area(self):
        """Return the surface area of this Cylinder instance"""
        
        return 2*(self.l_charged*self.w_charged)
    
    def get_volume(self):
        
        return self.l*self.h+self.w
    
    def get_name(self):
        
        return self.name
    
    def get_thickness(self):
        
        return self.h
    
    def get_geo_file_name(self):
        
        return self.geo_file_name

    def get_charge_positions_normals(self, charge_to_side_tol, hex_pack_dist):
    
        """
        Returns new charge locations on a cubic geometry and normal to the dielectric surface at 
        that points which is used to trnsform the initial pqr.
        """    
        n_half_seg_hor, n_seg_ver = get_flat_membrane_info(self.l, self.w, charge_to_side_tol, hex_pack_dist)
        l = n_half_seg_hor*hex_pack_dist*np.sqrt(3)/2
        w = n_seg_ver*hex_pack_dist
        n_points_hor = int(n_half_seg_hor/2) + 1
        n_points_ver = n_seg_ver + 1

        l_inner= np.linspace(-l/2, l/2 - hex_pack_dist * np.sqrt(3)/2, n_points_hor)
        l_outer= np.linspace(-l/2 + hex_pack_dist * np.sqrt(3)/2, l/2, n_points_hor)
        w_outer = np.linspace(-w/2, w/2, n_points_ver)
        w_inner = np.linspace(-w/2+hex_pack_dist/2, w/2-hex_pack_dist/2, n_points_ver-1)

        points_outer = np.meshgrid(l_outer, w_outer)
        points_outer = np.transpose(np.array(points_outer),(2, 1,0))
        points_outer = np.concatenate((points_outer, np.ones((points_outer.shape[0],points_outer.shape[1],1))*self.h),2)
        points_outer = points_outer.reshape((points_outer.shape[0]*points_outer.shape[1],3)).T

        points_inner = np.meshgrid(l_inner, w_inner)
        points_inner = np.transpose(np.array(points_inner),(2, 1,0))
        points_inner = np.concatenate((points_inner, np.ones((points_inner.shape[0],points_inner.shape[1],1))*self.h),2)
        points_inner = points_inner.reshape((points_inner.shape[0]*points_inner.shape[1],3)).T

        points = np.hstack((points_outer, points_inner)).T
        grads = np.zeros((points.shape))
        grads[:,2] = 1

        top_array = np.hstack((points, grads))

        top = pd.DataFrame({'X' :top_array[:,0],'Y' :top_array[:,1],'Z' :top_array[:,2],\
                            'nX':top_array[:,3],'nY':top_array[:,4],'nZ':top_array[:,5]})

        bot = top.copy()
        bot.loc[:,["Z"]] = 0
        top = translate_df_to(top, self.center)
        bot = translate_df_to(bot, self.center)
        
        return top, bot

    def write_gmsh_info(self, st, charge_to_side_tol, hex_pack_dist):
        
        #st = mesh.get_stern_thickness(mesh.interface_type)
        n_half_seg_hor, n_seg_ver = get_flat_membrane_info(self.l, self.w, charge_to_side_tol, hex_pack_dist)
        l_new = n_half_seg_hor*hex_pack_dist*np.sqrt(3)/2
        w_new = n_seg_ver*hex_pack_dist
        setattr(self,'l_mesh',l_new+charge_to_side_tol*2)                                    
        setattr(self,'w_mesh',w_new+charge_to_side_tol*2)                                    
        setattr(self,'l_charged',l_new+hex_pack_dist*np.sqrt(3)/2)                                    
        setattr(self,'w_charged',w_new+hex_pack_dist)                                    
        info = """l = %7.5f;
w = %7.5f;
h = %7.5f;
st = %7.5f;
x_c = %7.5f;
y_c = %7.5f;
z_c = %7.5f;
Box(1) = {x_c-l/2-st, y_c-w/2-st, z_c-st, l+2*st, w+2*st, h+2*st};""" % (l_new+charge_to_side_tol*2, w_new+charge_to_side_tol*2,\
                                                                self.h, st, self.center[0],self.center[1],self.center[2])
                                                       
        return info
