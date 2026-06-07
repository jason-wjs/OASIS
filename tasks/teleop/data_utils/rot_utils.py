import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
def quatToEuler(quat):
    """ 将四元数转换为欧拉角(roll, pitch, yaw)。 """
    eulerVec = np.zeros(3)
    qw, qx, qy, qz = quat
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)
    else:
        eulerVec[1] = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    return eulerVec

def get_projected_gravity(quat):
    """Get projected gravity"""
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def quat_rotate_inverse(q, v):
    """
    将向量 v 以四元数 q 的逆旋转进行变换。  
    为保持一致，以下代码与原脚本中的实现相同。
    """
    q = np.asarray(q)
    v = np.asarray(v)

    q_w = q[:, -1]      # w
    q_vec = q[:, :3]    # x, y, z

    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * (2.0 * q_w)[:, np.newaxis]
    dot = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * (2.0 * dot)

    return a - b + c

def quat_rotate_inverse_torch(q, v, scalar_first=True):
    if scalar_first:
        q = q[..., [1, 2, 3, 0]]
    else:
        q = q[..., [0, 1, 2, 3]]
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def quat_rotate_inverse_np(q, v, scalar_first=True):
    q = np.asarray(q)
    v = np.asarray(v)
    if scalar_first:
        q = q[..., [1, 2, 3, 0]]
    else:
        q = q[..., [0, 1, 2, 3]]
    q_w = q[..., -1]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c

def euler_from_quaternion_torch(quat_angle, scalar_first=True):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    if scalar_first:
        quat_angle = quat_angle[..., [1, 2, 3, 0]]
    else:
        quat_angle = quat_angle[..., [0, 1, 2, 3]]
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def euler_from_quaternion_np(quat, scalar_first=True):
    if scalar_first:
        quat = quat[..., [1, 2, 3, 0]]
    else:
        quat = quat[..., [0, 1, 2, 3]]
    
    x = quat[:,0]; y = quat[:,1]; z = quat[:,2]; w = quat[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z


def quat_diff_np(q1, q2, scalar_first=True):
    # Ensure quaternions are numpy arrays
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Convert to scipy Rotation object (scalar-first)
    r1 = R.from_quat(q1, scalar_first=scalar_first)
    r2 = R.from_quat(q2, scalar_first=scalar_first)

    # Relative rotation
    r_rel = r2 * r1.inv()

    # Rotation vector (axis * angle)
    rotvec = r_rel.as_rotvec()  # returns angle * axis vector

    return rotvec


def get_relative_pos(ref_pos, init_pos, init_ori):
    # 确保输入是 numpy 数组
    ref_pos = np.asarray(ref_pos).flatten()
    init_pos = np.asarray(init_pos).flatten()
    init_ori = np.asarray(init_ori).flatten()

    # 1. 计算世界坐标系下的位移
    delta_pos_world = ref_pos - init_pos
    
    # 2. 构造初始旋转 (wxyz -> xyzw)
    init_rotation = R.from_quat([init_ori[1], init_ori[2], init_ori[3], init_ori[0]])
    
    # 3. 关键：将位移向量转到初始帧定义的坐标系下
    # 这一步能保证如果 Motion 第一帧是朝向世界 X，那么 relative_pos 就是相对于那个方向的位移
    relative_pos_local = init_rotation.inv().apply(delta_pos_world)
    
    # 保持高度为 ref 的世界高度或相对高度（根据你训练数据的定义）
    # 如果训练数据第一帧 z 是 0，这里建议用: relative_pos_local[2] = ref_pos[2] - init_pos[2]
    relative_pos_local[2] = ref_pos[2] 
    
    return relative_pos_local

def get_relative_ori(ref_ori, init_ori):
    ref_ori = np.asarray(ref_ori).flatten()
    init_ori = np.asarray(init_ori).flatten()

    # wxyz -> xyzw
    r_ref = R.from_quat([ref_ori[1], ref_ori[2], ref_ori[3], ref_ori[0]])
    r_init = R.from_quat([init_ori[1], init_ori[2], init_ori[3], init_ori[0]])

    # 直接各自取 yaw（在世界系）
    yaw_ref = r_ref.as_euler('xyz')[2]
    yaw_init = r_init.as_euler('xyz')[2]

    # 相对 yaw（包到 [-pi, pi]）
    rel_yaw = np.arctan2(
        np.sin(yaw_ref - yaw_init),
        np.cos(yaw_ref - yaw_init)
    )

    # 只用 yaw 构造四元数
    res_q = R.from_euler('z', rel_yaw).as_quat()

    # xyzw -> wxyz
    return np.array([res_q[3], res_q[0], res_q[1], res_q[2]])


def get_yaw_quat_only(q):
    """从四元数中只提取 Yaw 部分并返回新的四元数"""
    # sin(yaw/2) = z, cos(yaw/2) = w (简化版，假设主要是绕Z轴旋转)
    # 标准做法是转成 euler 再转回 quat
    import numpy as np
    # 计算 yaw: atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    yaw = np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
    return np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])


def quat_apply_torch(q, v):
    # q: [4], v: [T, N, 3]
    w = q[0]
    xyz = q[1:]
    # v' = v + 2 * xyz x (xyz x v + w * v)
    t = 2.0 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)

def apply_quat_transform_torch(q_offset, q_targets):
    """
    使用 PyTorch 批量将 q_offset 应用到 q_targets。
    
    Args:
        q_offset: 变换四元数，形状 [4] 或 [B, 4] 或 [T, N, 4]
        q_targets: 目标四元数序列，形状 [..., 4] (如 [T, N, 4])
    Returns:
        旋转后的四元数，形状与 q_targets 一致
    """
    # 确保是 torch.Tensor
    if not isinstance(q_offset, torch.Tensor):
        q_offset = torch.from_numpy(np.array(q_offset)).float()
    if not isinstance(q_targets, torch.Tensor):
        q_targets = torch.from_numpy(np.array(q_targets)).float()

    # 拆分分量 (w, x, y, z)
    w1, x1, y1, z1 = q_offset[..., 0], q_offset[..., 1], q_offset[..., 2], q_offset[..., 3]
    w2, x2, y2, z2 = q_targets[..., 0], q_targets[..., 1], q_targets[..., 2], q_targets[..., 3]

    # 四元数乘法公式: q_res = q_offset * q_targets
    res_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    res_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    res_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    res_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([res_w, res_x, res_y, res_z], dim=-1)


def quat_inv_numpy(q):
    """计算四元数的逆 (w, x, y, z)"""
    q = np.asarray(q)
    res = q.copy()
    res[..., 1:] *= -1.0  # 单位四元数的逆等于其共轭
    return res

def quat_mul_numpy(q1, q2):
    """批量四元数乘法 q1 * q2 (w, x, y, z)"""
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    res = np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], axis=-1)
    return res

def get_relative_quat_transform(q_from, q_to):
    """
    计算从 q_from 到 q_to 的相对变换 q_offset
    满足: q_to = q_offset * q_from
    """
    # q_offset = q_to * inv(q_from)
    return quat_mul_numpy(q_to, quat_inv_numpy(q_from))

def quat_apply_numpy(q, v):
    """
    使用四元数 q 旋转向量 v (支持批量)
    v 形状可以为 (3,) 或 (N, 3)
    """
    q = np.asarray(q)
    v = np.asarray(v)

    w = q[0]
    xyz = q[1:]

    # v' = v + 2 * xyz x (xyz x v + w * v)
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_slerp_numpy(q0, q1, alpha):
    """Shortest-path slerp between two wxyz quats at scalar alpha in [0,1]."""
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return out / max(np.linalg.norm(out), 1e-12)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * alpha
    sin_t0 = np.sin(theta_0)
    s0 = np.sin(theta_0 - theta) / sin_t0
    s1 = np.sin(theta) / sin_t0
    return s0 * q0 + s1 * q1


def quat_axis_angle_vel(cur_q, prev_q, dt):
    """World-frame angular velocity from two wxyz quats via axis-angle / dt."""
    cur64 = np.asarray(cur_q, dtype=np.float64)
    prev64 = np.asarray(prev_q, dtype=np.float64)
    inv_prev = np.array([prev64[0], -prev64[1], -prev64[2], -prev64[3]], dtype=np.float64)
    w1, x1, y1, z1 = cur64
    w2, x2, y2, z2 = inv_prev
    q_delta = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)
    if q_delta[0] < 0:
        q_delta = -q_delta  # shortest-path
    w_clamped = float(np.clip(q_delta[0], -1.0, 1.0))
    half_angle = float(np.arccos(w_clamped))
    sin_half = float(np.sin(half_angle))
    if sin_half > 1e-6:
        axis = q_delta[1:4] / sin_half
        angle = 2.0 * half_angle
        return (axis * angle / float(dt)).astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def quat_to_rot6d(q):
    """First two columns of the rotation matrix (flattened row-major), wxyz in."""
    w, x, y, z = q
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    return np.array([r00, r01, r10, r11, r20, r21], dtype=np.float32)




