"""
Extract full hand attachment structure from the USDA file.
Outputs: joint hierarchy, mimic joints, collision bodies, fixed joints (hand-to-wrist attachment),
and joint limits/drive properties.
"""
import re
import os

USDA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "human_readable_g1_hands.usda"
)

def extract_hand_structure(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # We'll collect:
    # 1. All joint definitions (revolute, fixed, prismatic) with their full prim paths
    # 2. Joint properties: body0, body1, localPos, axis, limits, drive, mimic
    # 3. Fixed joints (how hands attach to wrists)
    # 4. Collision prims under hand bodies
    # 5. References (what USD files are referenced for the hands)

    joints = []
    fixed_joints = []
    references = []
    collision_prims = []
    articulation_root_path = None

    current_joint = None
    current_path_parts = []
    depth = 0
    prim_stack = []  # (depth, name, type)

    in_joint = False
    joint_lines = []
    joint_start_depth = 0

    # Simpler approach: scan for specific patterns
    all_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    print(f"Loaded {len(all_lines)} lines from USDA\n")

    # === PASS 1: Find references (what hand USD files are loaded) ===
    print("=" * 80)
    print("1. USD REFERENCES (Hand assets loaded into the scene)")
    print("=" * 80)
    for i, line in enumerate(all_lines):
        if 'references' in line.lower() and ('.usd' in line.lower() or '.usda' in line.lower() or '.usdc' in line.lower()):
            # Look for context - what prim is this under?
            for j in range(max(0, i-5), i):
                if 'def ' in all_lines[j]:
                    print(f"  Prim: {all_lines[j].strip()}")
                    break
            print(f"  Reference: {line.strip()}")
            print()

    # === PASS 2: Find all revolute joints with their properties ===
    print("=" * 80)
    print("2. ALL REVOLUTE JOINTS (Full list with drive properties)")
    print("=" * 80)

    i = 0
    while i < len(all_lines):
        line = all_lines[i]

        # Detect joint definition
        m = re.match(r'\s*def\s+(PhysicsRevoluteJoint|PhysicsFixedJoint|PhysicsPrismaticJoint)\s+"([^"]+)"', line)
        if m:
            joint_type = m.group(1)
            joint_name = m.group(2)

            # Collect all lines until the matching closing brace
            brace_count = 0
            joint_block = []
            j = i
            while j < len(all_lines):
                jline = all_lines[j]
                clean = re.sub(r'".*?"', '""', jline)
                brace_count += clean.count('{') - clean.count('}')
                joint_block.append(jline)
                if brace_count <= 0 and j > i:
                    break
                j += 1

            block_text = ''.join(joint_block)

            # Extract properties
            props = {
                'name': joint_name,
                'type': joint_type,
                'body0': None, 'body1': None,
                'localPos0': None, 'localPos1': None,
                'localRot0': None, 'localRot1': None,
                'axis': None,
                'lower_limit': None, 'upper_limit': None,
                'stiffness': None, 'damping': None,
                'maxForce': None, 'maxVelocity': None,
                'mimic': False, 'mimic_ref': None, 'mimic_gearing': None, 'mimic_offset': None,
            }

            for bl in joint_block:
                if 'rel physics:body0' in bl:
                    props['body0'] = bl.split('=')[-1].strip().strip('>')
                if 'rel physics:body1' in bl:
                    props['body1'] = bl.split('=')[-1].strip().strip('>')
                if 'physics:localPos0' in bl:
                    props['localPos0'] = bl.split('=')[-1].strip()
                if 'physics:localPos1' in bl:
                    props['localPos1'] = bl.split('=')[-1].strip()
                if 'physics:localRot0' in bl:
                    props['localRot0'] = bl.split('=')[-1].strip()
                if 'physics:localRot1' in bl:
                    props['localRot1'] = bl.split('=')[-1].strip()
                if 'physics:axis' in bl:
                    props['axis'] = bl.split('=')[-1].strip().strip('"')
                if 'physics:lowerLimit' in bl:
                    props['lower_limit'] = bl.split('=')[-1].strip()
                if 'physics:upperLimit' in bl:
                    props['upper_limit'] = bl.split('=')[-1].strip()
                if 'drive:angular:physics:stiffness' in bl:
                    props['stiffness'] = bl.split('=')[-1].strip()
                if 'drive:angular:physics:damping' in bl:
                    props['damping'] = bl.split('=')[-1].strip()
                if ':maxForce' in bl and 'drive' in bl:
                    props['maxForce'] = bl.split('=')[-1].strip()
                if 'maxJointVelocity' in bl:
                    props['maxVelocity'] = bl.split('=')[-1].strip()
                if 'PhysxMimicJointAPI' in bl:
                    props['mimic'] = True
                if 'physxMimicJoint:referenceJoint' in bl:
                    props['mimic_ref'] = bl.split('=')[-1].strip()
                if 'physxMimicJoint:gearing' in bl:
                    props['mimic_gearing'] = bl.split('=')[-1].strip()
                if 'physxMimicJoint:offset' in bl:
                    props['mimic_offset'] = bl.split('=')[-1].strip()

            if joint_type == 'PhysicsFixedJoint':
                fixed_joints.append(props)
            else:
                joints.append(props)

            i = j + 1
            continue
        i += 1

    # Print RIGHT hand joints
    print("\n--- RIGHT HAND JOINTS ---")
    print(f"{'Joint Name':<40} | {'Axis':<4} | {'Lower':>8} | {'Upper':>8} | {'Stiff':>8} | {'Damp':>8} | {'MaxF':>8} | {'Mimic?'}")
    print("-" * 120)
    for j in joints:
        if 'R_' in j['name']:
            mimic_str = f"YES gear={j['mimic_gearing']}" if j['mimic'] else "no"
            print(f"{j['name']:<40} | {(j['axis'] or '?'):<4} | {(j['lower_limit'] or 'N/A'):>8} | {(j['upper_limit'] or 'N/A'):>8} | {(j['stiffness'] or 'N/A'):>8} | {(j['damping'] or 'N/A'):>8} | {(j['maxForce'] or 'N/A'):>8} | {mimic_str}")

    print("\n--- LEFT HAND JOINTS ---")
    print(f"{'Joint Name':<40} | {'Axis':<4} | {'Lower':>8} | {'Upper':>8} | {'Stiff':>8} | {'Damp':>8} | {'MaxF':>8} | {'Mimic?'}")
    print("-" * 120)
    for j in joints:
        if 'L_' in j['name']:
            mimic_str = f"YES gear={j['mimic_gearing']}" if j['mimic'] else "no"
            print(f"{j['name']:<40} | {(j['axis'] or '?'):<4} | {(j['lower_limit'] or 'N/A'):>8} | {(j['upper_limit'] or 'N/A'):>8} | {(j['stiffness'] or 'N/A'):>8} | {(j['damping'] or 'N/A'):>8} | {(j['maxForce'] or 'N/A'):>8} | {mimic_str}")

    # Print G1 body joints (non-hand)
    print("\n--- G1 BODY JOINTS ---")
    print(f"{'Joint Name':<40} | {'Axis':<4} | {'Lower':>8} | {'Upper':>8} | {'Stiff':>8} | {'Damp':>8}")
    print("-" * 100)
    for j in joints:
        if 'R_' not in j['name'] and 'L_' not in j['name']:
            print(f"{j['name']:<40} | {(j['axis'] or '?'):<4} | {(j['lower_limit'] or 'N/A'):>8} | {(j['upper_limit'] or 'N/A'):>8} | {(j['stiffness'] or 'N/A'):>8} | {(j['damping'] or 'N/A'):>8}")

    # === PASS 3: Fixed joints (hand-to-wrist attachment) ===
    print("\n" + "=" * 80)
    print("3. FIXED JOINTS (Hand-to-wrist attachment)")
    print("=" * 80)
    for fj in fixed_joints:
        print(f"  Joint: {fj['name']}")
        print(f"    body0: {fj['body0']}")
        print(f"    body1: {fj['body1']}")
        print(f"    localPos0: {fj['localPos0']}")
        print(f"    localPos1: {fj['localPos1']}")
        print(f"    localRot0: {fj['localRot0']}")
        print(f"    localRot1: {fj['localRot1']}")
        print()

    # === PASS 4: Find collision prims under hand bodies ===
    print("=" * 80)
    print("4. COLLISION PRIMS (Bodies with collision geometry)")
    print("=" * 80)
    for i, line in enumerate(all_lines):
        if 'CollisionAPI' in line or 'collisionEnabled' in line:
            # Find the parent prim
            for j in range(max(0, i-10), i):
                if 'def ' in all_lines[j] and ('R_' in all_lines[j] or 'L_' in all_lines[j] or 'hand' in all_lines[j].lower()):
                    print(f"  {all_lines[j].strip()} -> line {i}: {line.strip()}")
                    break

    # === PASS 5: Find articulation root ===
    print("\n" + "=" * 80)
    print("5. ARTICULATION ROOT")
    print("=" * 80)
    for i, line in enumerate(all_lines):
        if 'ArticulationRootAPI' in line:
            for j in range(max(0, i-10), i):
                if 'def ' in all_lines[j]:
                    print(f"  Articulation root at: {all_lines[j].strip()}")
                    break
            print(f"  Line {i}: {line.strip()}")

    # === PASS 6: Mimic joint summary ===
    print("\n" + "=" * 80)
    print("6. MIMIC JOINT MAPPING (which joints follow which)")
    print("=" * 80)
    for j in joints:
        if j['mimic']:
            print(f"  {j['name']} -> mimics {j['mimic_ref']} (gearing: {j['mimic_gearing']}, offset: {j['mimic_offset']})")

    # Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    right_hand = [j for j in joints if 'R_' in j['name']]
    left_hand = [j for j in joints if 'L_' in j['name']]
    body_joints = [j for j in joints if 'R_' not in j['name'] and 'L_' not in j['name']]
    mimic_joints = [j for j in joints if j['mimic']]

    print(f"  Total revolute joints: {len(joints)}")
    print(f"  G1 body joints: {len(body_joints)}")
    print(f"  Right hand joints: {len(right_hand)}")
    print(f"  Left hand joints: {len(left_hand)}")
    print(f"  Mimic joints: {len(mimic_joints)}")
    print(f"  Fixed joints: {len(fixed_joints)}")

if __name__ == "__main__":
    extract_hand_structure(USDA_PATH)
