import mujoco
import numpy as np
import mujoco.viewer
import time
import os
import sys

print("Loading ONLY the right ORCA hand model...")
hand_path = "/home/tempo/.venv/lib/python3.13/site-packages/orca_sim/scene_right.xml"
spec = mujoco.MjSpec.from_file(hand_path)

wrist_joint = next(j for j in spec.joints if j.name == "right_wrist")
hand_body = wrist_joint.parent
while hand_body.parent is not None and hand_body.parent.name != "world":
    hand_body = hand_body.parent

hand_body_name = hand_body.name
hand_free_joint = hand_body.add_freejoint()
hand_free_joint.name = "hand_float_joint"
hand_body.gravcomp = 1.0  # Prevent floating hand from sagging under gravity

# Soften hand contact parameters and add rolling friction so the egg does not bounce away
for g in spec.geoms:
    if g.contype != 0 or g.conaffinity != 0:
        g.friction = [2.5, 1.0, 0.3]
        g.condim = 6
        g.solref = [0.015, 1.0]
        g.solimp = [0.90, 0.98, 0.002, 0.5, 2]

joint_names = [
    "right_wrist",
    "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
    "right_index_abd", "right_index_mcp", "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd", "right_ring_mcp", "right_ring_pip",
    "right_pinky_abd", "right_pinky_mcp", "right_pinky_pip"
]

# Only curl joints are incremented during adaptive grasp (abduction joints stay neutral/opposed)
finger_curl_joints = {
    "Thumb":  [1, 3, 4],    # thumb_mcp, thumb_pip, thumb_dip (thumb_abd=2 held in opposition)
    "Index":  [6, 7],       # index_mcp, index_pip
    "Middle": [9, 10],      # middle_mcp, middle_pip
    "Ring":   [12, 13],     # ring_mcp, ring_pip
    "Pinky":  [15, 16]      # pinky_mcp, pinky_pip
}

# =====================================================================
# --- AUTO-CALIBRATE HORIZONTAL ORIENTATION & EGG POSITION ---
# =====================================================================
temp_model = spec.compile()
temp_data = mujoco.MjData(temp_model)
temp_j_id = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, "hand_float_joint")
temp_qpos_adr = temp_model.jnt_qposadr[temp_j_id]

# 1. Detect valid flexion sign and joint limits for every joint
q_pregrasp = np.array([
    0.0,                    # wrist
    0.10, 0.90, 0.10, 0.10, # thumb: open, abd pre-opposed
    0.0,  0.10, 0.10,       # index open
    0.0,  0.10, 0.10,       # middle open
    0.0,  0.10, 0.10,       # ring open
    0.0,  0.10, 0.10        # pinky open
])

q_grasp_max = np.array([
    0.0,                    # wrist
    0.95, 0.95, 1.15, 1.05, # thumb fully curled + opposed
    0.0,  1.35, 1.30,       # index curled
    0.0,  1.35, 1.30,       # middle curled
    0.0,  1.35, 1.30,       # ring curled
    0.0,  1.35, 1.30        # pinky curled
])

flex_signs = np.ones(17)
for idx, name in enumerate(joint_names):
    jid = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if temp_model.jnt_limited[jid]:
        low, high = temp_model.jnt_range[jid]
        if abs(low) > abs(high):
            flex_signs[idx] = -1.0
            q_pregrasp[idx] = -q_pregrasp[idx]
            q_grasp_max[idx] = -q_grasp_max[idx]
        q_pregrasp[idx] = np.clip(q_pregrasp[idx], low, high)
        q_grasp_max[idx] = np.clip(q_grasp_max[idx], low, high)

# 2. Measure hand local axes in default pose so we can rotate it horizontally next to the egg
temp_data.qpos[temp_qpos_adr : temp_qpos_adr+3] = [0.0, 0.0, 0.25]
temp_data.qpos[temp_qpos_adr+3 : temp_qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(temp_model, temp_data)

mid_mcp_j = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, "right_middle_mcp")
mid_pip_j = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, "right_middle_pip")
thumb_mcp_j = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, "right_thumb_mcp")
pinky_mcp_j = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, "right_pinky_mcp")

p_mcp = temp_data.xanchor[mid_mcp_j].copy()
p_pip_open = temp_data.xanchor[mid_pip_j].copy()
p_thumb_mcp = temp_data.xanchor[thumb_mcp_j].copy()
p_pinky_mcp = temp_data.xanchor[pinky_mcp_j].copy()

e_finger = p_pip_open - p_mcp
e_finger /= np.linalg.norm(e_finger)

# Flex middle MCP slightly to measure exact palm-inward curl direction
temp_data.qpos[temp_model.jnt_qposadr[mid_mcp_j]] = 0.45 * flex_signs[9]
mujoco.mj_forward(temp_model, temp_data)
p_pip_flex = temp_data.xanchor[mid_pip_j].copy()

v_palm = (p_pip_flex - p_pip_open)
v_palm -= np.dot(v_palm, e_finger) * e_finger
e_palm = v_palm / np.linalg.norm(v_palm)

e_thumb = np.cross(e_finger, e_palm)
if np.dot(e_thumb, p_thumb_mcp - p_pinky_mcp) < 0:
    e_palm = -e_palm
    e_thumb = np.cross(e_finger, e_palm)

R_horiz = np.vstack([e_finger, e_palm, e_thumb]).astype(np.float64)
q_horiz = np.zeros(4, dtype=np.float64)
mujoco.mju_mat2Quat(q_horiz, R_horiz.flatten())

# 3. Preview the horizontal hand 65% closed to find the exact grasp pocket & safe table height
ref_z = 0.25
temp_data.qpos[:] = 0.0
temp_data.qpos[temp_qpos_adr : temp_qpos_adr+3] = [0.0, 0.0, ref_z]
temp_data.qpos[temp_qpos_adr+3 : temp_qpos_adr+7] = q_horiz
for idx, name in enumerate(joint_names):
    jid = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    temp_data.qpos[temp_model.jnt_qposadr[jid]] = q_pregrasp[idx] + 0.65 * (q_grasp_max[idx] - q_pregrasp[idx])
mujoco.mj_forward(temp_model, temp_data)

thumb_dip_j = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, "right_thumb_dip")
index_pip_j = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, "right_index_pip")
ring_pip_j = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, "right_ring_pip")

pocket_pos = np.mean([
    temp_data.xanchor[thumb_dip_j],
    temp_data.xanchor[index_pip_j],
    temp_data.xanchor[mid_pip_j],
    temp_data.xanchor[ring_pip_j]
], axis=0)

# Check lowest hand body Z in open horizontal pose so pinky clears the table top (z=0.02m)
for idx, name in enumerate(joint_names):
    jid = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    temp_data.qpos[temp_model.jnt_qposadr[jid]] = q_pregrasp[idx]
mujoco.mj_forward(temp_model, temp_data)

hand_body_id_temp = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_BODY, hand_body_name)
hand_body_z_vals = []
for b_id in range(1, temp_model.nbody):
    curr = b_id
    while curr != 0:
        if curr == hand_body_id_temp:
            hand_body_z_vals.append(temp_data.xpos[b_id][2])
            break
        curr = temp_model.body_parentid[curr]
min_hand_z_at_ref = min(hand_body_z_vals)

egg_radius = 0.025
table_top_z = 0.02
egg_rest_z = table_top_z + egg_radius  # 0.045 m

shift_z_for_egg = egg_rest_z - pocket_pos[2]
min_shift_z_for_table = (table_top_z + 0.015) - min_hand_z_at_ref
grasp_z = ref_z + max(shift_z_for_egg, min_shift_z_for_table)

hand_x_grasp = 0.0
hand_y_grasp = 0.0
egg_x = pocket_pos[0]
egg_y = pocket_pos[1]

approach_dist = 0.14
hand_y_start = hand_y_grasp - approach_dist
lift_z = grasp_z + 0.18

print(f"\n--- HORIZONTAL SIDE-GRASP COORDINATES ---")
print(f"Hand Start Pos:  X={hand_x_grasp:.3f}, Y={hand_y_start:.3f}, Z={grasp_z:.3f}")
print(f"Hand Grasp Pos:  X={hand_x_grasp:.3f}, Y={hand_y_grasp:.3f}, Z={grasp_z:.3f}")
print(f"Egg Spawn Pos:   X={egg_x:.3f}, Y={egg_y:.3f}, Z={egg_rest_z:.3f}")
print(f"-----------------------------------------\n")

# Add Table
table = spec.worldbody.add_body()
table.name = "table"
table.pos = [egg_x + 0.10, egg_y, 0.0]
table.add_geom(
    name="table_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.6, 0.6, table_top_z],
    rgba=[0.5, 0.3, 0.1, 1],
    friction=[2.0, 1.0, 0.3],
    condim=6
)

# Add Egg
egg = spec.worldbody.add_body()
egg.name = "egg"
egg.pos = [egg_x, egg_y, egg_rest_z]
egg.add_freejoint(name="egg_free_joint")
egg_geom = egg.add_geom(
    name="egg_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[egg_radius, 0.0, 0.0],
    mass=0.05,
    rgba=[1.0, 0.9, 0.8, 1],
    friction=[2.5, 1.0, 0.3],
    condim=6
)
egg_geom.solref = [0.015, 1.0]
egg_geom.solimp = [0.90, 0.98, 0.002, 0.5, 2]

model = spec.compile()
data = mujoco.MjData(model)

hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body_name)
hand_j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_float_joint")
hand_qpos_adr = model.jnt_qposadr[hand_j_id]
hand_dof_adr = model.jnt_dofadr[hand_j_id]

egg_j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "egg_free_joint")
egg_qpos_adr = model.jnt_qposadr[egg_j_id]
egg_dof_adr = model.jnt_dofadr[egg_j_id]
model.dof_damping[egg_dof_adr : egg_dof_adr+6] = 0.15

egg_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "egg_geom")
table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")

hand_geoms = set()
for i in range(model.ngeom):
    curr_body = model.geom_bodyid[i]
    while curr_body != 0:
        if curr_body == hand_body_id:
            hand_geoms.add(i)
            break
        curr_body = model.body_parentid[curr_body]

qposadrs, dofadrs, act_ids = [], [], []
for name in joint_names:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qposadrs.append(model.jnt_qposadr[j_id])
    dofadrs.append(model.jnt_dofadr[j_id])
    act_id = -1
    for i in range(model.nu):
        if model.actuator_trnid[i, 0] == j_id:
            act_id = i
            break
    act_ids.append(act_id)
    if act_id != -1:
        model.actuator_biastype[act_id] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[act_id, :] = 0.0
        model.actuator_gainprm[act_id, 0] = 1.0
        model.actuator_ctrlrange[act_id] = [-3.0, 3.0]

Kp = np.ones(17) * 10.0
Ki = np.ones(17) * 0.25
Kd = np.ones(17) * 0.6
integral_error = np.zeros(17)

dt = model.opt.timestep
current_x = hand_x_grasp
current_y = hand_y_start
current_z = grasp_z

data.qpos[hand_qpos_adr : hand_qpos_adr+3] = [current_x, current_y, current_z]
data.qpos[hand_qpos_adr+3 : hand_qpos_adr+7] = q_horiz
for idx, adr in enumerate(qposadrs):
    data.qpos[adr] = q_pregrasp[idx]
mujoco.mj_forward(model, data)

phase = 0
phase_step = 0
dy_per_step = 0.0007
dz_per_step = 0.0006
target_x_offset = 0.25
contact_counter = 0
grasp_hold_counter = 0

finger_progress = {f: 0.0 for f in finger_curl_joints}
q_desired = q_pregrasp.copy()
q_grasp_frozen = q_pregrasp.copy()
prev_base_pos = np.array([current_x, current_y, current_z])

stop_reason = "Simulation ended."
print("Starting Horizontal Side-Approach Pick and Place...")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(6000):
            if not viewer.is_running():
                stop_reason = "Simulation closed by user (Window closed)."
                break

            phase_step += 1
            hand_touches_egg = False
            egg_touches_table = False
            total_hand_force = 0.0
            finger_forces = {
                "Thumb": 0.0, "Index": 0.0, "Middle": 0.0,
                "Ring": 0.0, "Pinky": 0.0, "Palm": 0.0
            }

            for i in range(data.ncon):
                c = data.contact[i]
                if c.geom1 == egg_geom_id or c.geom2 == egg_geom_id:
                    other_geom = c.geom2 if c.geom1 == egg_geom_id else c.geom1

                    if other_geom in hand_geoms:
                        hand_touches_egg = True
                        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
                        body_id = model.geom_bodyid[other_geom]
                        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                        tag = (geom_name + " " + body_name).lower()

                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, i, force)
                        f_mag = np.linalg.norm(force[:3])
                        total_hand_force += f_mag

                        if "thumb" in tag: finger_forces["Thumb"] += f_mag
                        elif "index" in tag: finger_forces["Index"] += f_mag
                        elif "middle" in tag: finger_forces["Middle"] += f_mag
                        elif "ring" in tag: finger_forces["Ring"] += f_mag
                        elif "pinky" in tag: finger_forces["Pinky"] += f_mag
                        else: finger_forces["Palm"] += f_mag

                    if other_geom == table_geom_id:
                        egg_touches_table = True

            if phase == 0:  # Phase 0: Slide Horizontally Next to the Egg (+Y)
                q_desired = q_pregrasp.copy()
                integral_error[:] = 0.0

                if hand_touches_egg and (hand_y_grasp - current_y) < 0.04:
                    contact_counter += 1
                else:
                    contact_counter = 0

                if current_y >= hand_y_grasp or contact_counter >= 2:
                    print(f"\nHorizontal approach complete (Y={current_y:.3f}). Starting Adaptive Grasp!")
                    grasp_hold_counter = 0
                    phase, phase_step = 1, 0
                else:
                    current_y += dy_per_step

            elif phase == 1:  # Phase 1: Adaptive Grasp (supports both thumb-finger & palm-finger grasps)
                opposing_base_f = finger_forces["Thumb"] + finger_forces["Palm"]
                fingers_f = (
                    finger_forces["Index"] + finger_forces["Middle"]
                    + finger_forces["Ring"] + finger_forces["Pinky"]
                )
                fingers_touching_count = sum(
                    1 for name in ["Index", "Middle", "Ring", "Pinky"] if finger_forces[name] > 0.2
                )

                if phase_step % 20 == 0:
                    print(
                        f"Grasping... | T:{finger_forces['Thumb']:.2f} P:{finger_forces['Palm']:.2f} "
                        f"| I:{finger_forces['Index']:.2f} M:{finger_forces['Middle']:.2f} "
                        f"R:{finger_forces['Ring']:.2f} Pk:{finger_forces['Pinky']:.2f} "
                        f"| Total:{total_hand_force:.2f} N"
                    )

                # Count consecutive steps of firm caging/pinching contact
                has_firm_contact = (
                    (opposing_base_f > 0.2 and fingers_f > 0.3)
                    or (total_hand_force > 1.0 and fingers_touching_count >= 2)
                )
                if has_firm_contact:
                    grasp_hold_counter += 1
                else:
                    grasp_hold_counter = 0

                all_curled = all(p >= 0.95 for p in finger_progress.values())

                # Transition once firm grasp holds for 25 steps OR fingers finish wrapping around the egg
                if (grasp_hold_counter >= 25 and phase_step >= 80) or (all_curled and hand_touches_egg and phase_step >= 120):
                    print(f">>> Grasp secured (Total force={total_hand_force:.2f} N). Locking pose to lift! <<<")
                    q_grasp_frozen = q_desired.copy()
                    for finger, indices in finger_curl_joints.items():
                        p_lock = min(finger_progress[finger] + 0.10, 1.0)
                        for idx in indices:
                            q_grasp_frozen[idx] = q_pregrasp[idx] + p_lock * (q_grasp_max[idx] - q_pregrasp[idx])
                    phase, phase_step = 2, 0

                elif phase_step > 600:
                    if hand_touches_egg:
                        print(">>> Max grasp time reached with contact active. Attempting lift... <<<")
                        q_grasp_frozen = q_desired.copy()
                        phase, phase_step = 2, 0
                    else:
                        print("FAILURE: No contact with the egg after 600 steps.")
                        stop_reason = "Grasp failed: Hand did not touch the egg."
                        break

                else:
                    for finger, indices in finger_curl_joints.items():
                        f = finger_forces[finger]
                        if f < 0.8:
                            finger_progress[finger] = min(finger_progress[finger] + 0.006, 1.0)
                        elif f > 4.5:
                            finger_progress[finger] = max(finger_progress[finger] - 0.002, 0.0)

                        for idx in indices:
                            q_desired[idx] = (
                                q_pregrasp[idx]
                                + finger_progress[finger] * (q_grasp_max[idx] - q_pregrasp[idx])
                            )

                    q_desired[0] = 0.0  # Keep wrist straight

            elif phase == 2:  # Phase 2: Settle Hold
                q_desired = q_grasp_frozen
                if phase_step >= 60:
                    print("Hold stable. Lifting egg!")
                    phase, phase_step = 3, 0

            elif phase == 3:  # Phase 3: Lift Up (+Z)
                q_desired = q_grasp_frozen
                current_z += dz_per_step
                if current_z >= lift_z:
                    current_z = lift_z
                    egg_z = data.qpos[egg_qpos_adr + 2]
                    if egg_z > 0.08:
                        print(f"SUCCESS! Egg lifted to Z = {egg_z:.3f}")
                    else:
                        print(f"WARNING: Egg slipped during lift (Egg Z={egg_z:.3f}).")
                    phase, phase_step = 4, 0

            elif phase == 4:  # Phase 4: Move Sideways (+X)
                q_desired = q_grasp_frozen
                progress = min(phase_step / 300.0, 1.0)
                current_x = hand_x_grasp + (target_x_offset * progress)
                if phase_step >= 300:
                    phase, phase_step = 5, 0

            elif phase == 5:  # Phase 5: Lower to Place (-Z)
                q_desired = q_grasp_frozen
                current_x = hand_x_grasp + target_x_offset
                if (current_z < grasp_z + 0.03 and egg_touches_table) or current_z <= grasp_z:
                    print(f"Egg placed on table at Z = {current_z:.3f}. Releasing!")
                    phase, phase_step = 6, 0
                else:
                    current_z -= dz_per_step

            elif phase == 6:  # Phase 6: Release & Retract Horizontally (-Y)
                current_x = hand_x_grasp + target_x_offset
                progress = min(phase_step / 100.0, 1.0)
                q_desired = q_grasp_frozen + (q_pregrasp - q_grasp_frozen) * progress
                if phase_step > 60:
                    current_y -= dy_per_step
                if phase_step > 140:
                    stop_reason = "Success: Horizontal Pick and Place completed."
                    break

            # Apply base pose AND real Cartesian velocity so friction carries the egg
            base_pos = np.array([current_x, current_y, current_z])
            base_vel = (base_pos - prev_base_pos) / dt
            prev_base_pos = base_pos.copy()

            data.qpos[hand_qpos_adr : hand_qpos_adr+3] = base_pos
            data.qpos[hand_qpos_adr+3 : hand_qpos_adr+7] = q_horiz
            data.qvel[hand_dof_adr : hand_dof_adr+3] = base_vel
            data.qvel[hand_dof_adr+3 : hand_dof_adr+6] = 0.0

            q_current = np.array([data.qpos[adr] for adr in qposadrs])
            qd_current = np.array([data.qvel[adr] for adr in dofadrs])

            err = q_desired - q_current
            integral_error = np.clip(integral_error + err * dt, -0.5, 0.5)

            tau = (Kp * err) + (Ki * integral_error) - (Kd * qd_current)
            tau = np.clip(tau, -2.5, 2.5)

            for i, act_id in enumerate(act_ids):
                if act_id != -1:
                    data.ctrl[act_id] = tau[i]

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.005)

except KeyboardInterrupt:
    stop_reason = "Simulation interrupted by user (Ctrl+C)."

finally:
    print(f"\nReason for closing simulation: {stop_reason}")
    sys.stdout.flush()
    os._exit(0)
