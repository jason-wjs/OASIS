import math
import torch
import random
import numpy as np
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_from_euler_xyz
from pxr import Gf, Sdf, UsdGeom, Vt, Usd, UsdShade, UsdLux
from isaaclab.envs import ManagerBasedEnv


class TextureHelper:

    _dumped_assets: set[str] = set()
    _warned_pools: set[str] = set()

    @staticmethod
    def find_material_name(prim) -> str | None:
        """Walk up the prim tree until we hit a UsdShade.Material; return
        its prim name (e.g. 'Wall_Mat'). None if no Material ancestor."""
        p = prim.GetParent()
        while p.IsValid():
            if p.IsA(UsdShade.Material):
                return p.GetName()
            p = p.GetParent()
        return None

    @staticmethod
    def get_current_diffuse(material_prim) -> str | None:
        """Find the diffuse_texture asset path currently bound to a Material.

        Walks every descendant Shader of the given Material and returns the
        first input matching ``*diffuse_texture*``. Returns None if no shader
        has a diffuse_texture set.
        """
        if not material_prim or not material_prim.IsValid():
            return None
        for child in Usd.PrimRange(material_prim):
            if not child.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(child)
            for inp in shader.GetInputs():
                if "diffuse_texture" not in inp.GetFullName():
                    continue
                val = inp.Get()
                if val is None:
                    continue
                # USD asset paths come back as Sdf.AssetPath; the resolved
                # path is the most useful form, falling back to the authored
                # path if no resolver entry is set.
                try:
                    resolved = val.resolvedPath or val.path
                except AttributeError:
                    resolved = str(val)
                return str(resolved) if resolved else None
        return None

    # ── Pool spec resolution ──────────────────────────────────────────────

    @classmethod
    def resolve_pool(cls, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            out: list[str] = []
            for v in value:
                out.extend(cls.resolve_pool(v))
            return out
        if not isinstance(value, str):
            return []

        import os
        import glob

        # Remote URL: pass through unchanged. USD's Sdf.AssetPath plus the
        # registered resolvers (Omniverse client, http resolver, etc.) handle
        # remote fetching natively when the path is fed back to a shader.
        _URL_SCHEMES = ("http://", "https://", "omniverse://", "s3://", "file://")
        if value.startswith(_URL_SCHEMES):
            return [value]

        # Glob pattern: any string containing wildcard chars
        if any(ch in value for ch in "*?["):
            files = sorted(glob.glob(value, recursive=True))
            if not files and value not in cls._warned_pools:
                print(f"[randomize_texture] WARNING: glob {value!r} matched 0 files")
                cls._warned_pools.add(value)
            return files

        # Directory: collect common image extensions inside (non-recursive)
        if os.path.isdir(value):
            files: list[str] = []
            for pat in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
                files.extend(glob.glob(os.path.join(value, pat)))
            files = sorted(set(files))
            if not files and value not in cls._warned_pools:
                print(f"[randomize_texture] WARNING: directory {value!r} contains 0 png/jpg files")
                cls._warned_pools.add(value)
            return files

        # Single existing file
        if os.path.isfile(value):
            return [value]

        if value not in cls._warned_pools:
            print(f"[randomize_texture] WARNING: texture pool {value!r} is neither a directory, glob, nor existing file")
            cls._warned_pools.add(value)
        return []

    # ── Material discovery / dump ─────────────────────────────────────────

    @classmethod
    def dump_asset_materials(
        cls,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        env_id: int = 0,
        save_path: str | None = None,
    ) -> dict:

        asset = env.scene[asset_cfg.name]
        stage = env.scene.stage
        root_path = asset.prim_paths[env_id]
        root_prim = stage.GetPrimAtPath(root_path)

        if not root_prim.IsValid():
            print(f"[dump_asset_materials] WARNING: root prim invalid at {root_path}")
            return {"materials": {}, "meshes": {}}

        materials: dict[str, dict] = {}
        meshes: dict[str, list[str]] = {}

        # Pass 1: enumerate every Material under <root>/Looks/ or <root>/Materials/
        for scope_name in ("Looks", "Materials"):
            scope_prim = stage.GetPrimAtPath(f"{root_path}/{scope_name}")
            if scope_prim.IsValid():
                for prim in Usd.PrimRange(scope_prim):
                    if prim.IsA(UsdShade.Material):
                        materials[prim.GetName()] = {
                            "path":   str(prim.GetPath()),
                            "meshes": [],
                        }

        # Pass 2: walk meshes, resolve their bound material, cross-link both
        # directions. Uses MaterialBindingAPI which respects USD inheritance
        # and collection bindings.
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            binding_api = UsdShade.MaterialBindingAPI(prim)
            bound, _ = binding_api.ComputeBoundMaterial()
            if not bound:
                continue
            mat_name = bound.GetPrim().GetName()
            mesh_name = prim.GetName()
            mesh_path = str(prim.GetPath())

            if mat_name in materials:
                materials[mat_name]["meshes"].append(mesh_path)
            else:
                # Material lives outside <root>/Looks/ — still record it
                materials[mat_name] = {
                    "path":   str(bound.GetPrim().GetPath()),
                    "meshes": [mesh_path],
                }
            meshes.setdefault(mesh_name, []).append(mat_name)

        cls._print_dump(asset_cfg.name, root_path, materials, stage)

        if save_path:
            import json
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({"materials": materials, "meshes": meshes},
                          f, indent=2, ensure_ascii=False)
            print(f"  → full report saved to {save_path}\n")

        return {"materials": materials, "meshes": meshes}

    @classmethod
    def _print_dump(cls, asset_name: str, root_path: str,
                    materials: dict[str, dict], stage) -> None:
        """Render the dump_asset_materials result as a clean banner."""
        used_materials = {n: i for n, i in materials.items() if i["meshes"]}
        unused_count = len(materials) - len(used_materials)

        HEADER_WIDTH = 72
        MAX_MESHES_SHOWN = 6

        print()
        print("═" * HEADER_WIDTH)
        print(f"  asset:  {asset_name}")
        print(f"  root:   {root_path}")
        print(f"  bound:  {len(used_materials)} material(s)     unused: {unused_count} (skipped)")
        print("═" * HEADER_WIDTH)

        if not used_materials:
            print("  (no bound materials)")
            print("═" * HEADER_WIDTH)
            return

        sorted_mats = sorted(used_materials.items(),
                             key=lambda kv: -len(kv[1]["meshes"]))
        max_name = max(len(n) for n in used_materials.keys())
        name_col = max(max_name + 2, 32)
        prefix = root_path.rstrip("/") + "/"

        for mat_name, info in sorted_mats:
            n = len(info["meshes"])
            label = "mesh" if n == 1 else "meshes"
            print()
            print(f"  ▸ {mat_name:<{name_col}} {n:>3} {label}")

            mat_prim = stage.GetPrimAtPath(info["path"])
            current = cls.get_current_diffuse(mat_prim)
            tex_label = current.split("/")[-1] if current else "(constant color, no diffuse_texture)"
            print(f"      texture:  {tex_label}")

            # Strip the asset root prefix so mesh paths read as relative
            rel_paths = [mp[len(prefix):] if mp.startswith(prefix) else mp
                         for mp in info["meshes"]]
            shown = rel_paths[:MAX_MESHES_SHOWN]
            for i, rp in enumerate(shown):
                tag = "meshes:  " if i == 0 else "         "
                print(f"      {tag}{rp}")
            if n > MAX_MESHES_SHOWN:
                print(f"               … {n - MAX_MESHES_SHOWN} more")

        print()
        print("═" * HEADER_WIDTH)

    @classmethod
    def maybe_dump_once(cls, env: ManagerBasedEnv,
                        asset_cfg: SceneEntityCfg, env_id: int) -> None:
        if asset_cfg.name in cls._dumped_assets:
            return
        cls._dumped_assets.add(asset_cfg.name)
        try:
            cls.dump_asset_materials(env, asset_cfg, env_id=env_id)
        except Exception as e:
            print(f"[randomize_texture] dump_asset_materials failed: {e}")


# ── Randomization entry points (called by Isaac Lab EventTerm) ───────────────

def randomize_texture(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        material_names: list[str],
        event_name: str,
        texture_groups: dict[str, list[str] | str] | None = None,
        verbose: bool = False,
):
    asset = env.scene[asset_cfg.name]
    stage = env.scene.stage

    # First-call discovery dump (one-shot per asset, controlled by verbose).
    if verbose:
        TextureHelper.maybe_dump_once(env, asset_cfg, env_id=int(env_ids[0].item()))

    # Normalize each pool spec — caller can pass dirs / globs / files / lists.
    groups: dict[str, list[str]] = {}
    if texture_groups:
        for k, v in texture_groups.items():
            groups[k] = TextureHelper.resolve_pool(v)

    default_pool: list[str] = groups.get("*", [])

    select_all = "*" in material_names
    for env_id in env_ids:
        # 获取当前环境的根路径
        root_path = asset.prim_paths[env_id]
        for scope_name in ("Looks", "Materials"):
            scope_prim = stage.GetPrimAtPath(f"{root_path}/{scope_name}")
            if not scope_prim.IsValid():
                continue
            for prim in Usd.PrimRange(scope_prim):
                shader = UsdShade.Shader(prim)
                for shader_input in shader.GetInputs():
                    name = shader_input.GetFullName()
                    # 逻辑：如果是全选模式，或者当前节点名/父节点名在列表里
                    is_target = select_all or (prim.GetName() in material_names)

                    if is_target:
                        if "diffuse_texture" in name:
                            # Per-material pool: walk up to find the Material
                            # this Shader belongs to, then look up its name.
                            material_name = TextureHelper.find_material_name(prim)
                            pool = groups.get(material_name, default_pool) if material_name else default_pool
                            if pool:
                                shader_input.Set(random.choice(pool))
                        elif "roughness" in name:
                            type_name = shader_input.GetTypeName()
                            if type_name in [Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Double]:
                                if "influence" in name:
                                    shader_input.Set(random.uniform(0.0, 0.5))
                                else:
                                    shader_input.Set(random.uniform(0.1, 0.65))
                        elif "texture_rotate" in name:
                            shader_input.Set(random.uniform(0.0, 45.0))
                        elif "texture_translate" in name:
                            t = random.uniform(0.1, 1.0)
                            shader_input.Set(Gf.Vec2f(t, t))
                        # elif "texture_scale" in name:
                        #     s = random.uniform(0.5, 1.0)
                        #     shader_input.Set(Gf.Vec2f(s, s))
                        elif "project_uvw" in name:
                            shader_input.Set(bool(random.random() < 0.9))
                        elif "metallic_constant" in name:
                            shader_input.Set(random.uniform(0.25, 1.0))



def randomize_light(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        domelight_cfg: SceneEntityCfg,
):
    asset = env.scene[asset_cfg.name]
    domelight = env.scene[domelight_cfg.name]
    stage = env.scene.stage

    domelight_path = "/World/light"
    domelight_prim = stage.GetPrimAtPath(domelight_path)
    domelight_prim.GetAttribute("inputs:intensity").Set(np.random.uniform(1000, 3000))
    domelight_prim.GetAttribute("inputs:enableColorTemperature").Set(True)
    domelight_prim.GetAttribute("inputs:colorTemperature").Set(np.random.uniform(4500, 6500))
    domelight_prim.GetAttribute("inputs:color").Set(
            (
                np.random.uniform(0.85, 1.0),
                np.random.uniform(0.85, 1.0),
                np.random.uniform(0.85, 1.0),
            )
        )
    for env_id in env_ids:
        root_path = asset.prim_paths[env_id]
        light_path = f"{root_path}/Lights"
        light_prim = stage.GetPrimAtPath(light_path)
        if not light_prim.IsValid():
            continue

        for prim in Usd.PrimRange(light_prim):
            if prim.HasAPI(UsdLux.LightAPI):
                intensity = prim.GetAttribute("inputs:intensity")
                prim.GetAttribute("inputs:enableColorTemperature").Set(True)
                color_temperature = prim.GetAttribute("inputs:colorTemperature")
                color = prim.GetAttribute("inputs:color")
                if intensity.IsValid():
                    intensity.Set(max(0.0, np.random.uniform(20000, 200000)))
                if color_temperature.IsValid():
                    color_temperature.Set(np.random.uniform(4500, 6500))
                if color.IsValid():
                    new_color = Gf.Vec3f(
                        np.random.uniform(0.85, 1.0),
                        np.random.uniform(0.85, 1.0),
                        np.random.uniform(0.85, 1.0)
                    )
                    color.Set(new_color)
            else:
                continue


# ── Camera randomization ────────────────────────────────────────────────────

def _set_camera_transform(prim, pos_xyz, quat_wxyz):
    """Write xformOp:translate + xformOp:orient on a camera prim."""
    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        name = op.GetOpName()
        if name == "xformOp:translate":
            translate_op = op
        elif name == "xformOp:orient":
            orient_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])))

    if orient_op is None:
        orient_op = xformable.AddOrientOp()
    w, x, y, z = (float(v) for v in quat_wxyz)
    if orient_op.GetTypeName() == Sdf.ValueTypeNames.Quatd:
        orient_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        orient_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def randomize_cameras(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    camera_specs: dict,
    verbose: bool = False,
):
    """Per-reset randomization of camera mount + focal length.

    ``camera_specs`` maps ``scene_entity_name -> {pos_noise, rot_noise_deg,
    focal_noise_ratio}``. Each perturbation is sampled relative to the
    nominal offset from ``sensor.cfg`` (not from the currently-written USD
    state), so samples never compound across resets.

    - ``pos_noise``: scalar or 3-tuple, meters, uniform symmetric range
    - ``rot_noise_deg``: scalar or 3-tuple, degrees, uniform symmetric XYZ
      Euler delta applied as ``q_new = q_nominal * q_delta``
    - ``focal_noise_ratio``: scalar, focal sampled as ``nominal * (1 ± r)``
    """
    stage = env.scene.stage
    for cam_name, spec in camera_specs.items():
        sensor = env.scene[cam_name]
        cfg = sensor.cfg
        prim_path_tmpl = cfg.prim_path
        nominal_pos = tuple(cfg.offset.pos)
        nominal_rot = tuple(cfg.offset.rot)
        nominal_focal = float(cfg.spawn.focal_length)

        pos_noise = spec.get("pos_noise", 0.0)
        if np.isscalar(pos_noise):
            pos_noise = (float(pos_noise),) * 3
        rot_noise_deg = spec.get("rot_noise_deg", 0.0)
        if np.isscalar(rot_noise_deg):
            rot_noise_deg = (float(rot_noise_deg),) * 3
        focal_noise_ratio = float(spec.get("focal_noise_ratio", 0.0))

        for env_id in env_ids:
            eid = int(env_id.item()) if hasattr(env_id, "item") else int(env_id)
            # isaaclab already expands {ENV_REGEX_NS} to ``/World/envs/env_.*``
            # at spawn time, so we substitute the regex wildcard here.
            prim_path = prim_path_tmpl.replace("env_.*", f"env_{eid}")
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                if verbose:
                    print(f"[randomize_cameras] WARNING: prim invalid at {prim_path}")
                continue

            dpos = tuple(np.random.uniform(-n, n) for n in pos_noise)
            new_pos = tuple(nominal_pos[i] + dpos[i] for i in range(3))

            deuler_rad = [math.radians(np.random.uniform(-n, n)) for n in rot_noise_deg]
            dquat = quat_from_euler_xyz(
                torch.tensor(deuler_rad[0]),
                torch.tensor(deuler_rad[1]),
                torch.tensor(deuler_rad[2]),
            )
            new_quat = quat_mul(torch.tensor(nominal_rot, dtype=dquat.dtype), dquat).tolist()

            new_focal = nominal_focal * (1.0 + np.random.uniform(-focal_noise_ratio, focal_noise_ratio))

            _set_camera_transform(prim, new_pos, new_quat)
            focal_attr = prim.GetAttribute("focalLength")
            if focal_attr.IsValid():
                focal_attr.Set(float(new_focal))



def randomize_robot_texture(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    asset = env.scene[asset_cfg.name] # 这是一个 Articulation 对象
    stage = env.scene.stage

    for env_id in env_ids:
        robot_root_path = asset.cfg.prim_path.replace(".*", str(env_id.item()))
        robot_prim = stage.GetPrimAtPath(robot_root_path)
        if not robot_prim.IsValid():
            robot_root_path = f"/World/envs/env_{env_id.item()}/{asset_cfg.name}"
            robot_prim = stage.GetPrimAtPath(robot_root_path)

        if not robot_prim.IsValid():
            continue

        for prim in Usd.PrimRange(robot_prim):
            if prim.IsA(UsdShade.Shader):
                shader = UsdShade.Shader(prim)
                for shader_input in shader.GetInputs():
                    name = shader_input.GetFullName()
                    if "metallic" in name:
                        shader_input.Set(random.uniform(0.8, 1.0))
                    elif "specular" in name:
                        shader_input.Set(random.uniform(0.0, 0.8))
                    elif "roughness" in name:
                        shader_input.Set(random.uniform(0.4, 0.8))
