import numpy as np
#import cupy as cp
from scipy import signal
from scipy.spatial import cKDTree
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont

#def generate_kernels(size=3):
#    

def calculate_vorticity(field_matrix):
    """
    Calculates vorticity of a phase angle matrix.  
    This essentially computes the line integral of the angle distances
    around a plaquette.
    Parameters
    ----------
    field_matrix: matrix of the phase angles

    Return:
    a matrix containing the normalized vorticity.
    """
    
    val = np.pad(field_matrix, (1,1), mode='wrap')
    """
    k1  = np.array([[ 0,-1, 1],
                    [ 0, 0, 0],
                    [ 0, 0, 0]])
    k2  = np.array([[-1, 1, 0],
                    [ 0, 0, 0],
                    [ 0, 0, 0]])
    k3  = np.array([[ 1, 0, 0],
                    [-1, 0, 0],
                    [ 0, 0, 0]])
    k4  = np.array([[ 0, 0, 0],
                    [ 1, 0, 0],
                    [-1, 0, 0]])
    k5  = np.array([[ 0, 0, 0],
                    [ 0, 0, 0],
                    [ 1,-1, 0]])
    k6  = np.array([[ 0, 0, 0],
                    [ 0, 0, 0],
                    [ 0, 1,-1]])
    k7  = np.array([[ 0, 0, 0],
                    [ 0, 0,-1],
                    [ 0, 0, 1]])
    k8  = np.array([[ 0, 0,-1],
                    [ 0, 0, 1],
                    [ 0, 0, 0]])
    """


    k1 = np.array([[0,0,-1, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    k2 = np.array([[0,-1, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])
    k3 = np.array([[-1, 1, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])
    k4 = np.array([[ 1, 0, 0, 0],
                   [-1, 0, 0, 0],
                   [ 0, 0, 0, 0],
                   [ 0, 0, 0, 0]])
    k5 = np.array([[ 0, 0, 0, 0],
                   [ 1, 0, 0, 0],
                   [-1, 0, 0, 0],
                   [0, 0, 0, 0]])
    k6 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [1, 0, 0, 0],
                   [-1, 0, 0, 0]])
    k6 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [1,-1, 0, 0]])
    k7 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 1, -1, 0]])
    k8 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 1,-1]])
    k9 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0,-1],
                   [0, 0, 0, 1]])
    k10 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, -1],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
    k11 = np.array([[0, 0, 0,-1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    
    kernels = [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11]
    
    # conpute the vorticity
    def normalize_vorticity(vorticity):
        """
        Normalizes the vorticity submatrix between -pi and pi

        This is because the angle differences which is calculated by the convolution is \in [-2pi, 2pi].
        """
        vorticity = np.where(vorticity <= -np.pi, vorticity+2*np.pi, vorticity)
        vorticity = np.where(np.pi <= vorticity, vorticity-2*np.pi, vorticity)
        return vorticity
    
    vorticity = list(map(lambda k: normalize_vorticity(signal.fftconvolve(val, k[::-1, ::-1], mode='valid')), kernels))

    # sum across the row vector
    vorticity = np.sum(vorticity, axis=0)

    return vorticity

def find_clusters(locations, radius, boundary):
    """
    Looks for clusters of vortexes.  It is possible for the vortex core
    to be quite large.  This looks for a cluster of vortexes within a radius.

    Parameters
    ----------
    locations : listlike that contains the locations of vortexes in 2d for example: [(x1, y1), (x2,y2)]
    radius    : the radius at which you will merge cores
    boundary  : the boundaries for periodic boundary conditions.

    Return: dict containing the clusters and members.
    """
    cluster_id = 0
    clusters = {}
    remaining_locations = np.copy(locations)
    visited = {}

    for coord in locations:
        coord_tuple = (coord[0], coord[1])
        if coord_tuple in visited:
            continue


        cluster_idx = cKDTree(remaining_locations, boxsize=boundary).query_ball_point(coord, radius)
        if len(cluster_idx) != 0:
            cluster_coords = remaining_locations[cluster_idx]
            # store the cluster by cluster_id
            clusters[cluster_id] = cluster_coords
            cluster_id += 1
            # now remove the points that we have already processed
            visited[coord_tuple] = True
            for idx in cluster_idx:
                coord_tuple_temp = (remaining_locations[idx][0], remaining_locations[idx][1])
                visited[coord_tuple_temp] = True

    return clusters

def periodic_1d_average(coordinates, boundary):
    """
    Finds the centroid in 1d for a list of coordinates.  This method respects periodic boundary conditions

    Parameters
    ----------
    coordinates : listlike containing the coordinates.  Example: [1,2,3,4,5]
    boundary : tuple containing the boundary for periodic boundary conditions

    Return: a float containing the centroid
    """
    # check the distance if we shift things to the left side over to the right side of the boxs

    # check the distance span to see if we need to correct for periodic boundaries
    dist = np.max(coordinates) - np.min(coordinates)
    if dist <= boundary / 2:
        #we are good, just do a normal average
        return np.average(coordinates)

    # ok, that didn't work, so lets try correcting
    left_avg = np.average(np.where(coordinates < boundary / 2, coordinates + boundary, coordinates))
    right_avg = np.average(np.where(coordinates >= boundary / 2, coordinates - boundary, coordinates))

    # now see which of these centroids is inside the box.
    if left_avg >= 0 and left_avg < boundary:
        return left_avg
    elif right_avg >= 0 and right_avg < boundary:
        return right_avg
    else:
        assert False    
    
def find_centroids(cluster_dict, boundary):
    """
    Finds the centroids in a cluster dict.  The cluster dict should be formatted such that

    {cluster_id: [(x,y), (x2,y2)], ....}

    Parameters
    ----------
    cluster_dict : a dictionary containing the cluster_id and a listlike of coordinates
    boundary : the boundary with regard to periodic boundary conditions

    Return : average locations in a list.
    """
    avg_locs = []
    for k, v in cluster_dict.items():
        # find the center of the points in v
        d1 = v[:,0]
        # find the centroid of the first dimension
        avg_d1 = periodic_1d_average(d1, boundary[0])

        # find the centroid of the second dimension
        d2 = v[:,1]
        avg_d2 = periodic_1d_average(d2, boundary[1])
        avg_locs.append([avg_d1, avg_d2])
        #avg_locs.append(np.average(v, axis=0))

    return avg_locs
                
def find_vortexes(phase_angles, radius, boundary=(256,256)):
    """
    Finds the vortex locations given a matrix of phase angles.

    Parameters
    ----------
    phase_angles : an NxM matrix of phase angles.
    radius : the radius within which we will assume duplicates.
    boundary : the boundary of the torus (periodic boundaries)

    Return : positive vortexes, negative_vortexes
    """
    # calculate vorticity
    vorticity = calculate_vorticity(phase_angles)

    # renormalize between -1 and 1
    normalized_vorticity = vorticity / (2*np.pi)

    # split this into pos and neg vortexes
    pos_locs = np.argwhere((normalized_vorticity <= 1.001) & (normalized_vorticity >= 0.999) |
                           (normalized_vorticity <= 2.001) & (normalized_vorticity >= 1.999))
    #breakpoint()
    neg_locs = np.argwhere((normalized_vorticity >= -1.001) & (normalized_vorticity <= -0.999) |
                           (normalized_vorticity >= -2.001) & (normalized_vorticity <= -1.999))

    pos_clusters = find_clusters(pos_locs, radius, boundary)
    neg_clusters = find_clusters(neg_locs, radius, boundary)

    # average the clusters.
    pos_centroids = find_centroids(pos_clusters, boundary)
    neg_centroids = find_centroids(neg_clusters, boundary)
            
    # now that we've found both sets of centroids, return them
    return pos_centroids, neg_centroids

def create_field_image(phase_angles):
    """
    Turns the phase angles into an hsv image in PIL

    Parameters
    ----------
    phase_angles : a matrix of the phase angles in an NxM Matrix (there is no time coordinate).

    Return: Image object.
    """
    # normalize the angles between 0 and 255
    cmap = plt.cm.get_cmap('hsv')
    normalized_angles = (phase_angles + np.pi) / (np.pi*2)
    colorized_angles = np.uint8(cmap(normalized_angles)*255)
    im = Image.fromarray(colorized_angles)
    return im
    

def create_overlay_field_image(phase_angles, pos_centroids, neg_centroids, rescale=1):
    """
    Turns the phase angles into an hsv image in PIL with charges overlaying the image.

    Parameters
    ----------
    phase_angles : a matrix of the phase angles in an NxM Matrix with no time coordinate)
    pos_centroids : the positive centroids in a List-Like
    neg_centroids : the negative centroids in a List-Like

    Return
    ------
    im : The Image without overlays
    overlay : the Image with charge overlays
    combined_image : both combined in a new image of size (X*rescale*2+10, Y*rescale)    
    """
    im = create_field_image(phase_angles)
    rescaled_x = phase_angles.shape[0]*rescale
    rescaled_y = phase_angles.shape[1]*rescale
    
    im = im.resize((rescaled_x, rescaled_y))
    #breakpoint()
    overlayed = im.copy()
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMonoBold.ttf", 11)
    d = ImageDraw.Draw(overlayed)
    
    def draw_charge(im_draw, x, y, sign, invert=False):
        x = x*rescale
        y = y*rescale
        inner = (255,255,255)
        outer = (0,0,0)
        if invert:
            inner = (0,0,0)
            outer = (255,255,255)
            
        im_draw.text((int(y+1), int(x)), sign, font=fnt, fill=inner, anchor='mm')
        im_draw.text((int(y-1), int(x)), sign, font=fnt, fill=inner, anchor='mm')
        im_draw.text((int(y), int(x+1)), sign, font=fnt, fill=inner, anchor='mm')
        im_draw.text((int(y), int(x-1)), sign, font=fnt, fill=inner, anchor='mm')
        im_draw.text((int(y), int(x)), sign, font=fnt, fill=outer, anchor='mm')

    for pos in pos_centroids:
        x_val = int(np.round(pos[0]))
        y_val = int(np.round(pos[1]))
        
        draw_charge(d, x_val, y_val, "+")

    for neg in neg_centroids:
        x_val = int(np.round(neg[0]))
        y_val = int(np.round(neg[1]))
        
        draw_charge(d, x_val, y_val, "-", invert=True)

    # now combine the two images
    combined_image = Image.new('RGB', (rescaled_x*2+10, rescaled_y))
    combined_image.paste(im, (0, 0))
    combined_image.paste(overlayed, (rescaled_x+10, 0))

    return im, overlayed, combined_image
    



    
    
