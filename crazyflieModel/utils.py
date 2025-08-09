import numpy as np


def get_rotation_matrix_zyx(phi=0, theta=0, psi=0):
    R_x=np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
    R_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    R_z=np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
    R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))

    return R_matrix


def get_rotation_matrix_zxy(phi=0, theta=0, psi=0):
    R_x=np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
    R_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    R_z=np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
    R_matrix=np.matmul(R_z,np.matmul(R_x,R_y))

    return R_matrix


def get_transfer_matrix_zyx(phi=0, theta=0):
    """
    This matrix multiplied by
    :param phi:
    :param theta:
    :return:
    """
    transfer_matrix = np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
    ])

    return transfer_matrix


def get_transfer_matrix_zxy(phi=0, theta=0):
    """
    This matrix multiplied by
    :param phi:
    :param theta:
    :return:
    """
    transfer_matrix = np.array([
        [np.cos(theta), 0.0, -np.cos(phi) * np.sin(theta)],
        [0, 1.0, np.sin(phi)],
        [np.sin(theta), 0.0, np.cos(phi) * np.cos(theta)]
    ])

    return transfer_matrix


def tj_from_line_vectorized(start_pos, end_pos, time_ttl, t_c_array):
    t_c_array = np.asarray(t_c_array)
    v_max = (end_pos - start_pos) * 2 / time_ttl

    # Allocate output arrays
    pos = np.zeros((len(t_c_array), 3))
    vel = np.zeros((len(t_c_array), 3))
    acc = np.zeros((len(t_c_array), 3))  # always zeros

    first_half = (t_c_array >= 0) & (t_c_array < time_ttl / 2)
    second_half = ~first_half

    # First half computations
    t_c1 = t_c_array[first_half][:, None]
    vel[first_half] = v_max * t_c1 / (time_ttl / 2)
    pos[first_half] = start_pos + t_c1 * vel[first_half] / 2

    # Second half computations
    t_c2 = t_c_array[second_half][:, None]
    vel[second_half] = v_max * (time_ttl - t_c2) / (time_ttl / 2)
    pos[second_half] = end_pos - (time_ttl - t_c2) * vel[second_half] / 2

    return pos, vel, acc

