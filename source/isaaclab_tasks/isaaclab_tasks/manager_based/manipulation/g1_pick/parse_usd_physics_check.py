import re
import os

def scan_usd_for_physx_best_practices(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    print(f"🔍 Scanning USD for PhysX Best Practices: {file_path}\n")

    # Track scope
    path_stack = []
    current_depth = 0
    
    # Store extracted info
    joints_info = {}
    collision_filters = []
    articulation_self_collisions = None

    # Regex patterns
    def_pattern = re.compile(r'^\s*def\s+([A-Za-z0-9_]+)\s+"([^"]+)"')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip strings for brace counting so it doesn't get confused by string literals
                clean_line = re.sub(r'".*?"', '""', line) 
                
                # Check for Prim Definition
                match = def_pattern.match(line)
                if match:
                    prim_type = match.group(1)
                    prim_name = match.group(2)
                    path_stack.append(prim_name)
                    
                    if prim_type in ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"]:
                        current_prim_path = "/" + "/".join(path_stack)
                        joints_info[current_prim_path] = {
                            "mimic_api": False,
                            "reference_joint": None,
                            "gearing": None,
                            "maxForce": None,
                            "maxVelocity": None,
                            "stiffness": None,
                            "damping": None
                        }
                
                current_prim_path = "/" + "/".join(path_stack) if path_stack else ""

                # Check for Attributes if we are inside a tracked joint
                if current_prim_path in joints_info:
                    if "PhysxMimicJointAPI" in line:
                        joints_info[current_prim_path]["mimic_api"] = True
                    if "physxMimicJoint:referenceJoint" in line:
                        joints_info[current_prim_path]["reference_joint"] = line.split("=")[-1].strip()
                    if "physxMimicJoint:gearing" in line:
                        joints_info[current_prim_path]["gearing"] = line.split("=")[-1].strip()
                    if ":maxForce" in line:
                        joints_info[current_prim_path]["maxForce"] = line.split("=")[-1].strip()
                    if "maxJointVelocity" in line:
                        joints_info[current_prim_path]["maxVelocity"] = line.split("=")[-1].strip()
                    if ":stiffness" in line:
                        joints_info[current_prim_path]["stiffness"] = line.split("=")[-1].strip()
                    if ":damping" in line:
                        joints_info[current_prim_path]["damping"] = line.split("=")[-1].strip()

                # Check for Collision Filters anywhere
                if "filteredPairs" in line:
                    if current_prim_path not in collision_filters:
                        collision_filters.append(current_prim_path)

                # Check for Self Collisions on Articulation Root
                if "enabledSelfCollisions" in line:
                    articulation_self_collisions = line.split("=")[-1].strip()

                # Update Depth and Stack dynamically
                open_braces = clean_line.count('{')
                close_braces = clean_line.count('}')
                
                for _ in range(open_braces):
                    current_depth += 1
                    
                for _ in range(close_braces):
                    current_depth -= 1
                    if path_stack and current_depth < len(path_stack):
                        path_stack.pop()

    except Exception as e:
        print(f"Error parsing: {e}")
        return
        
    # --- REPORTING ---
    print("====== 1. ARTICULATION ROOT ======")
    print(f"Enabled Self Collisions: {articulation_self_collisions}\n")

    print("====== 2. COLLISION FILTERS (Filtered Pairs) ======")
    if not collision_filters:
        print("❌ NO COLLISION FILTERS FOUND. (Hand will explode if self-collisions are enabled)")
    else:
        print(f"✅ Found {len(collision_filters)} prims with filteredPairs defined.")
        for p in collision_filters[:5]:
            print(f"  - {p}")
        if len(collision_filters) > 5:
            print("  ... and more.")
    print()

    print("====== 3. HAND JOINT CONFIGURATIONS (Right Hand) ======")
    mimic_count = 0
    missing_limits = 0
    
    print(f"{'JOINT NAME':<40} | {'MIMIC?':<6} | {'GEAR':<10} | {'STIFFNESS':<10} | {'DAMPING':<10} | {'MAX FORCE':<10} | {'MAX VEL'}")
    print("-" * 120)
    
    for path, info in joints_info.items():
        # Filter for just the right hand to keep output clean
        if "R_" in path and "joint" in path.lower():
            is_mimic = "YES" if info["mimic_api"] else "NO"
            if info["mimic_api"]: mimic_count += 1
            
            gear = info["gearing"] if info["gearing"] else "N/A"
            stiff = info["stiffness"] if info["stiffness"] else "None"
            damp = info["damping"] if info["damping"] else "None"
            m_force = info["maxForce"] if info["maxForce"] else "None"
            m_vel = info["maxVelocity"] if info["maxVelocity"] else "None"
            
            if m_force == "None" or m_vel == "None":
                missing_limits += 1
            
            short_path = path.split("/")[-1]
            print(f"{short_path:<40} | {is_mimic:<6} | {gear:<10} | {stiff:<10} | {damp:<10} | {m_force:<10} | {m_vel}")

    print("\n====== SUMMARY ======")
    if mimic_count == 0:
        print("❌ WARNING: No PhysxMimicJointAPI found. The USD is missing hardware-level kinematic coupling.")
    else:
        print(f"✅ Found {mimic_count} joints with Mimic API applied.")
        
    if missing_limits > 0:
        print(f"❌ WARNING: {missing_limits} right hand joints are missing maxForce or maxVelocity limits.")

if __name__ == "__main__":
    TARGET_USD = "/home/daatsi-aeres/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/human_readable_g1_hands.usda"
    scan_usd_for_physx_best_practices(TARGET_USD)