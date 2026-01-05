import numpy as np
import trimesh
from PIL import Image
import os

# ==========================================
# CONSTANTS
# ==========================================

INPUT_IMAGE = "lroc_color_poles_2k.tif" 
OUTPUT_STL = "moon_lithophane_2.stl"
INVERT_IMAGE = True 

# Dimensions (mm)
SPHERE_DIAMETER = 150.0 
MIN_THICKNESS = 0.8     
MAX_THICKNESS = 3.2     

# Mounting Cylinder
CYLINDER_ID = 63.0      
CYLINDER_OD = 69.0       
CYLINDER_HEIGHT = 15.0   

# Flange Configuration
# FLANGE_DIAMETER behavior:
#   - If < CYLINDER_ID: Acts as "Hole Diameter" for an internal bottom cap.
#   - If >= CYLINDER_ID: Acts as "Outer Diameter" for an external flange.
# Set FLANGE_THICKNESS <= 0 to disable.
FLANGE_DIAMETER = 38.0   # Example: 40mm < 65mm, so this creates a CAP with a 40mm hole.
FLANGE_THICKNESS = 3.0   

# Holes
CYLINDER_HOLES = True    
CYLINDER_HOLE_DIA = CYLINDER_HEIGHT * 0.8
CYLINDER_HOLE_COUNT = int(3.14 * CYLINDER_OD * 0.5 / CYLINDER_HOLE_DIA) 

# Top Opening
TOP_HOLE_OD = 65.0       

# Resolution Settings
TARGET_WIDTH = 500           
CYLINDER_RESOLUTION = 128    

# ==========================================
# 1. ROBUST SPHERE GENERATION
# ==========================================

def create_lithophane_sphere():
    print(f"Loading image: {INPUT_IMAGE}")
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"{INPUT_IMAGE} not found.")

    img = Image.open(INPUT_IMAGE).convert('L')
    aspect = img.height / img.width
    new_h = int(TARGET_WIDTH * aspect)
    img = img.resize((TARGET_WIDTH, new_h), Image.Resampling.LANCZOS)
    
    pixels = np.array(img).astype(float) / 255.0
    if INVERT_IMAGE:
        pixels = 1.0 - pixels
    
    rows, cols = pixels.shape
    print(f"Generating sphere topology ({cols}x{rows})...")

    # --- Geometry Setup ---
    R_base = (SPHERE_DIAMETER / 2.0) - MAX_THICKNESS
    overlap_margin = 1.0 
    cutoff_diameter = CYLINDER_OD - overlap_margin
    pole_angle = np.arcsin(cutoff_diameter / (2 * R_base))
    lat_cutoff = -np.pi/2 + pole_angle
    row_limit = int(rows * (1 - (lat_cutoff + np.pi/2)/np.pi))

    # --- Vertex Generation ---
    phi = np.linspace(0, 2*np.pi, cols, endpoint=False) 
    theta_start = np.pi/2 - (np.pi/rows) 
    theta = np.linspace(theta_start, lat_cutoff, row_limit)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    u_vec = np.linspace(0, 1, cols, endpoint=False)
    v_vec = np.linspace(1/rows, row_limit/rows, row_limit)
    px_vec = np.clip((u_vec * (cols - 1)).astype(int), 0, cols-1)
    py_vec = np.clip((v_vec * (rows - 1)).astype(int), 0, rows-1)
    px_grid, py_grid = np.meshgrid(px_vec, py_vec)
    intensities = pixels[py_grid, px_grid]

    thickness = MIN_THICKNESS + intensities * (MAX_THICKNESS - MIN_THICKNESS)
    r_outer = R_base + thickness
    r_inner = np.full_like(r_outer, R_base)

    def to_cart(r, th, ph):
        return np.column_stack((
            (r * np.cos(th) * np.cos(ph)).flatten(),
            (r * np.cos(th) * np.sin(ph)).flatten(),
            (r * np.sin(th)).flatten()
        ))

    verts_outer = to_cart(r_outer, theta_grid, phi_grid)
    verts_inner = to_cart(r_inner, theta_grid, phi_grid)

    pole_intensity = np.mean(pixels[0, :])
    pole_thick = MIN_THICKNESS + pole_intensity * (MAX_THICKNESS - MIN_THICKNESS)
    pole_outer = np.array([[0, 0, R_base + pole_thick]])
    pole_inner = np.array([[0, 0, R_base]])

    num_grid_verts = len(verts_outer)
    all_verts = np.vstack([verts_outer, pole_outer, verts_inner, pole_inner])
    
    off_vo = 0
    off_po = num_grid_verts
    off_vi = num_grid_verts + 1
    off_pi = 2 * num_grid_verts + 1

    print("Stitching topology...")
    faces = []

    def get_idx(r, c, offset):
        return offset + r * cols + (c % cols)

    # 1. Grid Body
    for r in range(row_limit - 1):
        for c in range(cols):
            p1 = get_idx(r, c, off_vo)
            p2 = get_idx(r, c+1, off_vo)
            p3 = get_idx(r+1, c+1, off_vo)
            p4 = get_idx(r+1, c, off_vo)
            faces.append([p1, p4, p2])
            faces.append([p2, p4, p3])
            
            ip1 = get_idx(r, c, off_vi)
            ip2 = get_idx(r, c+1, off_vi)
            ip3 = get_idx(r+1, c+1, off_vi)
            ip4 = get_idx(r+1, c, off_vi)
            faces.append([ip1, ip2, ip4])
            faces.append([ip2, ip3, ip4])

    # 2. North Pole Cap
    for c in range(cols):
        p_pole = off_po
        p1 = get_idx(0, c, off_vo)
        p2 = get_idx(0, c+1, off_vo)
        faces.append([p_pole, p1, p2])
        
        ip_pole = off_pi
        ip1 = get_idx(0, c, off_vi)
        ip2 = get_idx(0, c+1, off_vi)
        faces.append([ip_pole, ip2, ip1])

    # 3. Bottom Rim
    last_r = row_limit - 1
    for c in range(cols):
        p1 = get_idx(last_r, c, off_vo)
        p2 = get_idx(last_r, c+1, off_vo)
        ip1 = get_idx(last_r, c, off_vi)
        ip2 = get_idx(last_r, c+1, off_vi)
        faces.append([p1, p2, ip1])
        faces.append([p2, ip2, ip1])

    print("Creating Trimesh object...")
    mesh = trimesh.Trimesh(vertices=all_verts, faces=faces)
    
    if not mesh.is_volume:
        print("Mesh not volume yet. Running Trimesh process...")
        mesh.process(validate=True)
        if not mesh.is_volume:
            trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)

    print(f"Sphere Valid Volume: {mesh.is_volume}")
    return mesh

# ==========================================
# 2. GENERATE ADAPTER & BOOLEANS
# ==========================================

def create_adapter():
    print("Generating Adapter...")
    
    R_base = SPHERE_DIAMETER/2.0 - MAX_THICKNESS
    cutoff_diameter = CYLINDER_OD - 2.0
    pole_angle = np.arcsin(cutoff_diameter / (2 * R_base))
    z_sphere_bottom = R_base * np.sin(-np.pi/2 + pole_angle)
    
    # Z-levels
    # Sphere bottom connects to Cylinder Top
    z_cyl_top = z_sphere_bottom 
    # Cylinder goes down CYLINDER_HEIGHT from there
    z_cyl_bot = z_cyl_top - CYLINDER_HEIGHT
    
    # 1. Main Tube Body
    # Center Z of the main tube
    z_center = z_cyl_top - (CYLINDER_HEIGHT / 2.0)
    # Extra length upwards to overlap into sphere
    cyl_height = CYLINDER_HEIGHT + 5.0 
    z_center_adj = z_center + 2.5 # Shift up by half the overlap
    
    main_cyl = trimesh.creation.cylinder(radius=CYLINDER_OD/2.0, height=cyl_height, sections=CYLINDER_RESOLUTION)
    main_cyl.apply_translation([0, 0, z_center_adj])
    
    adapter_body = main_cyl
    
    # 2. Flange Logic
    using_cap_mode = False
    
    if FLANGE_THICKNESS > 0:
        z_flange = z_cyl_bot - (FLANGE_THICKNESS / 2.0)
        
        if FLANGE_DIAMETER < CYLINDER_ID:
            print(f"Cap Mode Detected ({FLANGE_DIAMETER} < {CYLINDER_ID}). Creating internal cap.")
            using_cap_mode = True
            # In cap mode, the solid flange must fill the tube, so it uses CYLINDER_OD (flush)
            flange_radius = CYLINDER_OD / 2.0
        else:
            print(f"Flange Mode Detected ({FLANGE_DIAMETER} >= {CYLINDER_ID}). Creating external flange.")
            flange_radius = FLANGE_DIAMETER / 2.0
            
        flange = trimesh.creation.cylinder(radius=flange_radius, height=FLANGE_THICKNESS, sections=CYLINDER_RESOLUTION)
        flange.apply_translation([0, 0, z_flange])
        
        adapter_body = trimesh.boolean.union([adapter_body, flange], engine='manifold')
    else:
        print("Flange disabled.")

    # 3. Cutters
    cutters = []
    
    if using_cap_mode:
        # CAP MODE: 
        # 1. Main hole (CYLINDER_ID) stops at the top of the flange.
        # 2. Cap hole (FLANGE_DIAMETER) cuts through everything (including the flange).
        
        # Main ID Cutter: From Top down to Z_Cyl_Bottom
        # To be safe, we make it overlap top, but stop exactly at z_cyl_bot
        cut_height_main = cyl_height # Matches the tube
        z_cut_main = z_center_adj    # Matches the tube
        
        main_cut = trimesh.creation.cylinder(radius=CYLINDER_ID/2.0, height=cut_height_main, sections=CYLINDER_RESOLUTION)
        main_cut.apply_translation([0, 0, z_cut_main])
        cutters.append(main_cut)
        
        # Cap Hole Cutter: Through hole
        cap_cut_height = cyl_height + FLANGE_THICKNESS + 20.0
        cap_cut = trimesh.creation.cylinder(radius=FLANGE_DIAMETER/2.0, height=cap_cut_height, sections=CYLINDER_RESOLUTION)
        cap_cut.apply_translation([0, 0, z_center]) # Rough center, long enough to cover all
        cutters.append(cap_cut)
        
    else:
        # STANDARD MODE:
        # Single hole of CYLINDER_ID cuts through everything (Tube + Flange)
        cut_height = cyl_height + FLANGE_THICKNESS + 20.0
        inner_cut = trimesh.creation.cylinder(radius=CYLINDER_ID/2.0, height=cut_height, sections=CYLINDER_RESOLUTION)
        inner_cut.apply_translation([0, 0, z_center])
        cutters.append(inner_cut)
    
    # 4. Side Holes
    if CYLINDER_HOLES:
        print(f"Generating {CYLINDER_HOLE_COUNT} side hole cutters...")
        cutter_len = 20.0
        offset_dist = CYLINDER_OD / 2.0
        
        for i in range(CYLINDER_HOLE_COUNT):
            angle = (2 * np.pi * i) / CYLINDER_HOLE_COUNT
            
            hole_cut = trimesh.creation.cylinder(radius=CYLINDER_HOLE_DIA/2.0, height=cutter_len, sections=CYLINDER_RESOLUTION)
            
            rot_y = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
            hole_cut.apply_transform(rot_y)
            
            hole_cut.apply_translation([offset_dist, 0, 0])
            
            z_rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
            hole_cut.apply_transform(z_rot)
            
            z_hole = z_sphere_bottom - (CYLINDER_HEIGHT / 2.0)
            hole_cut.apply_translation([0, 0, z_hole])
            
            cutters.append(hole_cut)

    # Boolean Difference
    adapter_final = trimesh.boolean.difference([adapter_body] + cutters, engine='manifold')
    
    return adapter_final

def create_top_hole_cutter():
    if TOP_HOLE_OD <= 0:
        return None
    print("Generating Top Hole Cutter...")
    cutter = trimesh.creation.cylinder(radius=TOP_HOLE_OD/2.0, height=SPHERE_DIAMETER, sections=CYLINDER_RESOLUTION)
    cutter.apply_translation([0, 0, SPHERE_DIAMETER/2.0])
    return cutter

def main():
    try:
        sphere = create_lithophane_sphere()
    except Exception as e:
        print(f"Sphere Generation Failed: {e}")
        return

    adapter = create_adapter()
    top_cutter = create_top_hole_cutter()
    
    print("Performing Boolean Union (Sphere + Adapter)...")
    try:
        combined = trimesh.boolean.union([sphere, adapter], engine='manifold')
    except Exception as e:
        print(f"Boolean Union failed: {e}")
        return

    if top_cutter:
        print("Performing Top Hole Cut...")
        try:
            combined = trimesh.boolean.difference([combined, top_cutter], engine='manifold')
        except Exception as e:
             print(f"Top Hole Cut failed: {e}")

    print(f"Saving to {OUTPUT_STL}...")
    combined.export(OUTPUT_STL)
    print("Success!")

if __name__ == "__main__":
    main()