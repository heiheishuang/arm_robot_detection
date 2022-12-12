import numpy as np
from deprecated.sphinx import deprecated


class Camera:

    def __init__(self, internal_mat, external_mat):
        self.internal_mat = internal_mat
        self.external_mat = external_mat
        self.homo_p2c = None
        self.homo_p2w = None
        self.homo_c2w = None
        self.projetction_mat = np.dot(self.internal_mat, self.external_mat)

    def load_param_two_mat(self, homo_p2c, homo_p2w):
        self.homo_p2c = homo_p2c
        self.homo_p2w = homo_p2w
        homo_c2p = np.linalg.inv(self.homo_p2c)
        self.homo_c2w = np.dot(self.homo_p2w, homo_c2p)

    def load_param_one_mat(self, homo_c2w):
        self.homo_c2w = homo_c2w

    def cal_world2pix(self, p_end):
        print("###########################")
        print("3D to 2D")
        tmp = (np.dot(self.projetction_mat, p_end.T))
        tmp = tmp / tmp[2]
        return tmp

    @deprecated(version="v1.0", reason="There's something wrong with this function")
    def cal_pix2world_by_projetction(self, p_camera):
        print("2D to 3D using projetction")
        projetction_mat_inv = np.linalg.pinv(self.projetction_mat)
        tmp = np.dot(projetction_mat_inv, p_camera.T)
        tmp = tmp / tmp[3]
        return tmp

    def cal_pix2world_by_homo(self, depth, p_camera):
        print("2D to 3D using homo")
        tmp = (np.dot(self.homo_c2w, p_camera))
        tmp = tmp / tmp[2]
        tmp[2] = depth
        return tmp

    @deprecated(version="v1.0", reason="There's something wrong with this function")
    def cal_pix2world_with_depth(self, depth, p_camera):
        print("2D to 3D using depth")
        f_x = self.internal_mat[0][0]
        f_y = self.internal_mat[1][1]
        c_x = self.internal_mat[0][2]
        c_y = self.internal_mat[1][2]
        z = depth
        x = (p_camera[0] - c_x) * z / f_x
        y = (p_camera[1] - c_y) * z / f_y
        result = np.array([x, y, z])
        return result


if __name__ == "__main__":
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
    print(camera.cal_pix2world_with_depth(depth=-67, p_camera=vec_camera))
    print(camera.cal_pix2world_by_projetction(p_camera=vec_camera))

    H_c2w = np.array([[0.0389321, 0.626301, -134.629],
                      [0.567981, -0.0602388, -387.209],
                      [0.000247696, -1.97379e-05, 1]])
    camera.load_param_one_mat(H_c2w)
    print(camera.cal_pix2world_by_homo(depth=-67, p_camera=vec_camera))
