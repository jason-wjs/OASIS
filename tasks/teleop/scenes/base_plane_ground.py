import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from .scene_factory import MotionsSceneCfgFactory
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

@MotionsSceneCfgFactory.register("plane")
class PlaneGroundSceneCfg(InteractiveSceneCfg):
    # 机器人初始位姿 (MuJoCo [w,x,y,z] quaternion)
    robot_init_pos = (-2.46, -1.1, 0.76)
    robot_init_rot = (0.7071, 0, 0, -0.7071)

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    

        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
        
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=3000.0),    # light intensity
    )