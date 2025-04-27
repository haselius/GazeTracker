import cv2
import numpy as np

def project_points_manual(pts_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Projects 3D points onto a 2D image plane using a given camera model.

    Args:
        pts_3d (numpy.ndarray): Array of 3D points (N x 3).
        rvec (numpy.ndarray): Rotation vector (3 x 1) in Rodrigues' format.
        tvec (numpy.ndarray): Translation vector (3 x 1).
        camera_matrix (numpy.ndarray): Intrinsic camera matrix (3 x 3).
        dist_coeffs (numpy.ndarray): Distortion coefficients (1 x 5).

    Returns:
        numpy.ndarray: Projected 2D points (N x 2).
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Camera Intrinsic Parameters
    f_x, f_y = camera_matrix[0, 0], camera_matrix[1, 1]  # Focal length
    c_x, c_y = camera_matrix[0, 2], camera_matrix[1, 2]  # Principal point
    gamma = camera_matrix[0, 1]  # Skew factor (typically 0)

    # Distortion coefficients
    k1, k2, p1, p2, k3 = dist_coeffs.ravel()

    projected_pts = []
    for pt in pts_3d:
        # Convert 3D point to camera coordinate system
        ### DO NOT DELETE THIS ###
        ### your code is here ###
        pass
        ### your code is here ###
        ### DO NOT DELETE THIS ###
        
        # Points in camera coordinate system (from homogeneous to Euclidean space)
        ### DO NOT DELETE THIS ###
        ### your code is here ###
        pass
        ### your code is here ###
        ### DO NOT DELETE THIS ###

        # Apply radial and tangential distortion
        ### DO NOT DELETE THIS ###
        ### your code is here ###
        pass
        ### your code is here ###
        ### DO NOT DELETE THIS ###

        # Convert to pixel coordinates
        ### DO NOT DELETE THIS ###
        ### your code is here ###
        pass
        x_s, y_s = None, None
        ### your code is here ###
        ### DO NOT DELETE THIS ###
                
        projected_pts.append((x_s, y_s))

    return np.array(projected_pts, dtype=np.float32)

def rotate_mesh(vertices, axis='x'):
    """
    Rotates a set of 3D vertices around a specified axis by 90 degrees.

    Args:
        vertices (numpy.ndarray): Array of 3D vertices (N x 3).
        axis (str, optional): Axis to rotate around ('x', 'y', or 'z'). Defaults to 'x'.

    Raises:
        ValueError: If the axis is not 'x', 'y', or 'z'.

    Returns:
        numpy.ndarray: Rotated vertices (N x 3).
    """
    theta = np.pi / 2  # 90-degree rotation
    
    if axis == 'x':
        ### DO NOT DELETE THIS ###
        ### your code is here ###
        rotation_matrix = np.array([
            #...
        ])
        ### your code is here ###
        ### DO NOT DELETE THIS ###
    elif axis == 'y':
        ### DO NOT DELETE THIS ###
        ### your code is here ###
        rotation_matrix = np.array([
            #...
        ])
        ### your code is here ###
        ### DO NOT DELETE THIS ###
    elif axis == 'z':
        ### DO NOT DELETE THIS ###
        ### your code is here ###
        rotation_matrix = np.array([
            #...
        ])
        ### your code is here ###
        ### DO NOT DELETE THIS ###
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    
    return rotated_vertices

def draw_faces(frame, projected_vertices, faces):
    """
    Draws polygonal faces on an image using projected 2D vertices.

    Args:
        frame (numpy.ndarray): Image to draw on.
        projected_vertices (numpy.ndarray): 2D projected vertices (N x 2).
        faces (list of lists): List of vertex indices defining faces.
    """
    for face in faces:
        pts = projected_vertices[face].astype(np.int32)
        ### DO NOT DELETE THIS ###
        ### your code is here ###
        pass
        ### your code is here ###
        ### DO NOT DELETE THIS ###

def draw_grid(image, color=(0, 255, 0), thickness=1):
    """
    Draws a grid overlay on the given image.

    Args:
        image (numpy.ndarray): The input image on which the grid will be drawn.
        color (tuple, optional): The color of the grid lines in BGR format. Defaults to (0, 255, 0) (green).
        thickness (int, optional): The thickness of the grid lines. Defaults to 1.

    Returns:
        numpy.ndarray: The image with the grid overlay.
    """
    h, w = image.shape[:2]

    grid_size_w = w / 20  
    grid_size_h = h / 10  
    
    grid_size_w = max(1, int(round(grid_size_w)))
    grid_size_h = max(1, int(round(grid_size_h)))
    
    # Draw vertical lines
    for x in range(0, w + 1, grid_size_w):  
        cv2.line(image, (x, 0), (x, h), color, thickness)

    # Draw horizontal lines
    for y in range(0, h + 1, grid_size_h):  
        cv2.line(image, (0, y), (w, y), color, thickness)

    return image
