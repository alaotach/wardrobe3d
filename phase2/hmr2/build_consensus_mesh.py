
from pathlib import Path

import numpy as np
import torch
import trimesh

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT


ROOT = Path(__file__).resolve().parent
PARAMS = ROOT / "outputs/hmr2_params/shape_consensus.npz"
OUTPUT = ROOT / "outputs/consensus_avatar"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Load the preliminary shared body shape.
with np.load(PARAMS) as data:
    betas_np = data["betas"].astype(np.float32)

# Load the SMPL body model.
model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
smpl = model.smpl.cpu().eval()

num_betas = int(smpl.num_betas)
if betas_np.size != num_betas:
    raise ValueError(
        f"Expected {num_betas} shape coefficients, got {betas_np.size}"
    )

betas = torch.from_numpy(betas_np).reshape(1, num_betas)

# Zero axis-angle rotations produce the SMPL rest/T-pose.
num_body_joints = int(smpl.NUM_BODY_JOINTS)

# HMR2 uses SMPLLayer, which expects rotation matrices:
# global_orient: [B, 1, 3, 3]
# body_pose:     [B, 23, 3, 3]

identity = torch.eye(3, dtype=torch.float32)

global_orient = identity.reshape(1, 1, 3, 3).clone()
body_pose = identity.reshape(1, 1, 3, 3).expand(
    1, num_body_joints, 3, 3
).contiguous()

with torch.no_grad():
    result = smpl(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
    )

vertices = result.vertices[0].cpu().numpy()
faces = np.asarray(smpl.faces)

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=False,
)

obj_path = OUTPUT / "consensus_avatar_tpose.obj"
glb_path = OUTPUT / "consensus_avatar_tpose.glb"

mesh.export(obj_path)
mesh.export(glb_path)

print("Shared-shape mesh generated.")
print("Vertices:", len(vertices))
print("Faces:", len(faces))
print("Dimensions (SMPL units):", mesh.extents)
print("OBJ:", obj_path)
print("GLB:", glb_path)
print("Betas:", np.round(betas_np, 4))
