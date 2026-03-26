import re
import sys
import os

def parse_usd_kinematics(file_path):
    # Regex to catch OpenUSD definitions like: def Xform "right_hand" or def PhysicsRevoluteJoint "R_index_joint"
    def_pattern = re.compile(r'^\s*def\s+([A-Za-z0-9_]+)\s+"([^"]+)"')
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    print(f"🔍 Scanning USD Kinematic Tree for: {file_path}\n")
    print("LEGEND:")
    print("  📦 Xform (Link/Body)")
    print("  🟢 PhysicsRevoluteJoint (Movable Hinge)")
    print("  🔴 PhysicsFixedJoint (Welded/Immovable)")
    print("-" * 60)

    depth = 0
    joint_count = 0

    try:
        # Read line-by-line to prevent RAM overflow on massive files
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                
                # Strip string literals temporarily to avoid miscounting braces inside strings
                clean_line = re.sub(r'".*?"', '""', line)
                
                # Check for an OpenUSD definition
                match = def_pattern.match(line)
                if match:
                    prim_type = match.group(1)
                    prim_name = match.group(2)
                    
                    # Filter only for kinematic structure and joints
                    if prim_type in ["Xform", "PhysicsRevoluteJoint", "PhysicsFixedJoint", "PhysicsPrismaticJoint"]:
                        indent = "  " * depth
                        
                        if prim_type == "PhysicsRevoluteJoint":
                            print(f"{indent}🟢 Revolute: {prim_name}")
                            joint_count += 1
                        elif prim_type == "PhysicsFixedJoint":
                            print(f"{indent}🔴 FIXED: {prim_name}")
                            joint_count += 1
                        elif prim_type == "PhysicsPrismaticJoint":
                            print(f"{indent}🟦 Prismatic: {prim_name}")
                            joint_count += 1
                        else:
                            print(f"{indent}📦 Body: {prim_name}")
                            
                # Track bracket depth to maintain the visual hierarchy
                depth += clean_line.count('{')
                depth -= clean_line.count('}')
                
                # Safety clamp for malformed lines
                if depth < 0:
                    depth = 0
                    
    except UnicodeDecodeError:
        print("\n❌ Error: File is not a valid ASCII text file. It might be a binary .usdc file.")
        print("You can convert binary USDs to ASCII using the 'usdcat' command-line tool included with Isaac Sim.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

    print("-" * 60)
    print(f"✅ Scan Complete. Found {joint_count} joints.")

if __name__ == "__main__":
    # Change this path to point to your massive USD file
    TARGET_USD = "/home/daatsi-aeres/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/human_readable_g1_hands.usda"
    parse_usd_kinematics(TARGET_USD)