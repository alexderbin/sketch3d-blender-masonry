"""Moss as volume (a flat layer read as green paint): instanced tufts on the same mask as the "moss" layer (bl_states) —
geometry nodes on the object: points on the faces with density × mask × "up", a tuft per point (its own size, rotation
around the normal), the material is the Moss003 moss scan (ambientCG) with box projection, each tuft has its own tint,
darker at the base. Where moss grows is decided by the masonry logic (the mask); volumetric moss is heavy — only near
the camera (the slab and the rows far away use a layer). The tufts are procedural (n shapes): a bumpy dome with a fine
fuzz of relief.

  import bl_moss as MS
  MS.moss3d(objs, mask="st_moss", density=1800, size=(0.012, 0.04), seed=1)   # → the objects get a moss3d modifier"""
import math
import os
import random
import sys

import bmesh
import bpy
from mathutils import Vector, noise

HERE = os.path.dirname(os.path.abspath(__file__))
from components.constructions.textures.states import bl_states as SS  # noqa: E402


def material():
    m = bpy.data.materials.get("sk_moss3d")
    if m is not None:
        return m
    m = bpy.data.materials.new("sk_moss3d"); nt = m.node_tree; N = SS.SF._N(nt)
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    mt = SS.meta("Moss003")
    tc = N.new("ShaderNodeTexCoord")
    t = N.new("ShaderNodeTexImage"); t.image = SS._img(mt["maps"]["color"], True); t.projection = "BOX"; t.projection_blend = 0.3
    N.put(t.inputs["Vector"], N.vmath("SCALE", tc.outputs["Object"], s=3.0))
    rnd = N.new("ShaderNodeObjectInfo")
    hue = N.new("ShaderNodeHueSaturation"); N.put(hue.inputs["Color"], t.outputs["Color"])
    N.put(hue.inputs["Hue"], N.math("ADD", 0.47, N.math("MULTIPLY", rnd.outputs["Random"], 0.06)))        # each tuft has its own tint
    N.put(hue.inputs["Value"], N.math("ADD", 0.7, N.math("MULTIPLY", rnd.outputs["Random"], 0.35)))
    N.put(hue.inputs["Saturation"], 1.15)
    sz = N.new("ShaderNodeSeparateXYZ"); N.put(sz.inputs[0], tc.outputs["Generated"])
    mr = N.new("ShaderNodeMapRange"); mr.clamp = True; N.put(mr.inputs["Value"], sz.outputs["Z"])
    mr.inputs["From Min"].default_value = 0.0; mr.inputs["From Max"].default_value = 0.6
    mr.inputs["To Min"].default_value = 0.35; mr.inputs["To Max"].default_value = 1.0                      # darker at the base
    cc = N.new("ShaderNodeCombineColor")
    for k in ("Red", "Green", "Blue"): N.put(cc.inputs[k], mr.outputs["Result"])
    N.put(b.inputs["Base Color"], N.mix_rgb("MULTIPLY", 1.0, hue.outputs["Color"], cc.outputs[0]))
    b.inputs["Roughness"].default_value = 0.88
    nz = N.new("ShaderNodeTexNoise", Scale=180.0, Detail=2.0); N.put(nz.inputs["Vector"], tc.outputs["Object"])
    bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.7; bp.inputs["Distance"].default_value = 0.002
    N.put(bp.inputs["Height"], nz.outputs["Fac"]); N.put(b.inputs["Normal"], bp.outputs["Normal"])
    return m


def tufts(n=6, seed=1):
    """n unit-size tufts (a dome ~0.45 high, bumps and fuzz), the sk_moss_tufts collection (not in the frame)."""
    col = bpy.data.collections.get("sk_moss_tufts")
    if col is not None and len(col.objects) >= n:
        return col
    col = col or bpy.data.collections.new("sk_moss_tufts")
    rng = random.Random(seed); mat = material()
    for i in range(n):
        bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=3, radius=0.5)
        off = Vector((rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-50, 50)))
        sx, sy = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -0.08], context="VERTS")
        for v in bm.verts:
            p = v.co.copy()
            lump = noise.fractal(p * 2.2 + off, 0.6, 2.0, 3)                          # tuft bumps
            fuzz = noise.noise(p * 40.0 + off) * 0.12                                  # fine fuzz
            k = 1.0 + 0.32 * lump + fuzz
            v.co = Vector((p.x * k * sx, p.y * k * sy, max(p.z, 0.0) * k * 0.9))
        me = bpy.data.meshes.new(f"sk_moss_tuft_{i}"); bm.to_mesh(me); bm.free()
        for p_ in me.polygons: p_.use_smooth = True
        me.materials.append(mat)
        col.objects.link(bpy.data.objects.new(f"sk_moss_tuft_{i}", me))
    return col


def _gn(col, mask, density, size, up):
    name = f"sk_moss3d_{mask}"
    g = bpy.data.node_groups.get(name)
    if g is not None:
        return g
    g = bpy.data.node_groups.new(name, "GeometryNodeTree")
    g.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    g.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nd, L = g.nodes, g.links
    gi = nd.new("NodeGroupInput"); go = nd.new("NodeGroupOutput")
    na = nd.new("GeometryNodeInputNamedAttribute"); na.data_type = "FLOAT"; na.inputs["Name"].default_value = mask
    nrm = nd.new("GeometryNodeInputNormal")
    dot = nd.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"; L.new(nrm.outputs["Normal"], dot.inputs[0]); dot.inputs[1].default_value = (0, 0, 1)
    upw = nd.new("ShaderNodeMapRange"); upw.clamp = True; L.new(dot.outputs["Value"], upw.inputs["Value"])
    upw.inputs["From Min"].default_value = -0.2; upw.inputs["From Max"].default_value = 0.8
    upw.inputs["To Min"].default_value = 1.0 - up; upw.inputs["To Max"].default_value = 1.0          # denser facing up, sparse on vertical surfaces
    fac0 = nd.new("ShaderNodeMath"); fac0.operation = "MULTIPLY"; L.new(na.outputs["Attribute"], fac0.inputs[0]); L.new(upw.outputs["Result"], fac0.inputs[1])
    pos = nd.new("GeometryNodeInputPosition")                                 # colonies: moss grows in patches with bare stone between them
    off = nd.new("ShaderNodeVectorMath"); off.operation = "ADD"; L.new(pos.outputs["Position"], off.inputs[0])
    off.inputs[1].default_value = SS._L.offset("moss")                          # the same noise as the "moss" layer: the tufts sit on its patches
    col_n = nd.new("ShaderNodeTexNoise"); col_n.inputs["Scale"].default_value = 3.5; col_n.inputs["Detail"].default_value = 3.0
    col_n.inputs["Roughness"].default_value = 0.55
    L.new(off.outputs[0], col_n.inputs["Vector"])
    patch = nd.new("ShaderNodeMapRange"); patch.clamp = True; L.new(col_n.outputs["Fac"], patch.inputs["Value"])
    patch.inputs["From Min"].default_value = 0.41; patch.inputs["From Max"].default_value = 0.61    # the same colony thresholds as the layer
    fac = nd.new("ShaderNodeMath"); fac.operation = "MULTIPLY"; L.new(fac0.outputs[0], fac.inputs[0]); L.new(patch.outputs["Result"], fac.inputs[1])
    dist = nd.new("GeometryNodeDistributePointsOnFaces"); dist.distribute_method = "RANDOM"
    L.new(gi.outputs[0], dist.inputs["Mesh"])
    dens = nd.new("ShaderNodeMath"); dens.operation = "MULTIPLY"; L.new(fac.outputs[0], dens.inputs[0]); dens.inputs[1].default_value = density
    L.new(dens.outputs[0], dist.inputs["Density"])                          # RANDOM: density is a field (mask × up × density)
    ci = nd.new("GeometryNodeCollectionInfo"); ci.inputs["Collection"].default_value = col
    ci.inputs["Separate Children"].default_value = True; ci.inputs["Reset Children"].default_value = True
    iop = nd.new("GeometryNodeInstanceOnPoints"); iop.inputs["Pick Instance"].default_value = True
    L.new(dist.outputs["Points"], iop.inputs["Points"]); L.new(ci.outputs[0], iop.inputs["Instance"])
    ri = nd.new("FunctionNodeRandomValue"); ri.data_type = "INT"; ri.inputs["Max"].default_value = len(col.objects) - 1
    ri.inputs["Seed"].default_value = 1                                     # Blender 5.2: the Random Value sockets are dynamic — Min, Max, ID, Seed → Value
    L.new(ri.outputs["Value"], iop.inputs["Instance Index"])
    al = nd.new("FunctionNodeAlignRotationToVector"); al.axis = "Z"; L.new(dist.outputs["Normal"], al.inputs["Vector"])
    ra = nd.new("FunctionNodeRandomValue"); ra.data_type = "FLOAT"; ra.inputs["Min"].default_value = 0.0; ra.inputs["Max"].default_value = 2 * math.pi
    ra.inputs["Seed"].default_value = 2
    spin = nd.new("FunctionNodeAxisAngleToRotation"); spin.inputs["Axis"].default_value = (0, 0, 1); L.new(ra.outputs["Value"], spin.inputs["Angle"])
    rot = nd.new("FunctionNodeRotateRotation"); rot.rotation_space = "LOCAL"
    L.new(al.outputs["Rotation"], rot.inputs["Rotation"]); L.new(spin.outputs["Rotation"], rot.inputs["Rotate By"])
    L.new(rot.outputs["Rotation"], iop.inputs["Rotation"])
    rs = nd.new("FunctionNodeRandomValue"); rs.data_type = "FLOAT"; rs.inputs["Min"].default_value = size[0]; rs.inputs["Max"].default_value = size[1]
    rs.inputs["Seed"].default_value = 3
    sc = nd.new("ShaderNodeMath"); sc.operation = "MULTIPLY"; L.new(rs.outputs["Value"], sc.inputs[0])
    big = nd.new("ShaderNodeMapRange"); big.clamp = True; L.new(na.outputs["Attribute"], big.inputs["Value"])
    big.inputs["To Min"].default_value = 0.5; big.inputs["To Max"].default_value = 1.15                  # where there is more moss, the tufts are larger
    L.new(big.outputs["Result"], sc.inputs[1])
    L.new(sc.outputs[0], iop.inputs["Scale"])
    jo = nd.new("GeometryNodeJoinGeometry"); L.new(gi.outputs[0], jo.inputs[0]); L.new(iop.outputs["Instances"], jo.inputs[0])
    L.new(jo.outputs[0], go.inputs[0])
    return g


def moss3d(objs, mask="st_moss", density=2600.0, size=(0.015, 0.05), up=0.88, seed=1):
    """Volumetric moss on objects by an attribute mask: the "moss3d" modifier (geometry nodes). density — tufts per m² at mask
    1."""
    col = tufts(seed=seed); g = _gn(col, mask, density, size, up); n = 0
    for ob in objs:
        if ob.type != "MESH" or mask not in ob.data.attributes: continue
        md = ob.modifiers.get("moss3d") or ob.modifiers.new("moss3d", "NODES"); md.node_group = g; n += 1
    return n
