import numpy as np
from Camera import Camera

C = np.array([[-0.000956971, 0.20166, 3.69457, 340.584],
              [0.251592, -0.085941, 3.16036, 225.981],
              [-7.52532e-05, -0.000234447, 0.013204, 1]])

internal = np.array([[20.2786, -0.713933, 279.439],
                     [0, 20.5392, 239.272],
                     [0, 0, 1]])
external = np.array([[-0.00145192, -0.013124, -0.000241302, -2.99245],
                     [-0.013126, 0.00145304, -4.90089e-05, 0.647105],
                     [7.52532e-05, 0.000234447, -0.013204, -1]])

H_p2c = np.array([[2.32388, -12.7171, 329.827],
                  [13.3201, 2.54399, 157.835],
                  [0.00022577, 0.00126462, 1]])

H_p2w = np.array([[7.79607, 0.859249, -21.2518],
                  [0.344352, -7.22112, -194.087],
                  [0.000711853, - 0.00208148, 1]])

vec_camera = np.array([330, 158, 1])
vec_end = np.array([-21.5215, -194.117, -67.1522, 1])

camera = Camera(internal, external)
camera.load_param_two_mat(H_p2c, H_p2w)

print(camera.cal_world2pix(p_end=vec_end))
print(camera.cal_pix2world_by_homo(depth=-67, p_camera=vec_camera))
# print(camera.cal_pix2world_with_depth(depth=-67, p_camera=vec_camera))
# print(camera.cal_pix2world_by_projetction(p_camera=vec_camera))

H_c2w = np.array([[0.0389321, 0.626301, -134.629],
                  [0.567981, -0.0602388, -387.209],
                  [0.000247696, -1.97379e-05, 1]])
camera.load_param_one_mat(H_c2w)
print(camera.cal_pix2world_by_homo(depth=-67, p_camera=vec_camera))
