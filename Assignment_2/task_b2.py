import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(obj_name="buddha", data_dir="../data_processed"):
    # Load lights
    lights_path = os.path.join(data_dir, "lights.txt")
    lights = np.loadtxt(lights_path)
    
    # Load mask
    mask_path = os.path.join(data_dir, obj_name, f"{obj_name}.mask.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask at {mask_path}")
        
    mask = mask > 127  # boolean mask
    
    # Load images
    images = []
    for i in range(len(lights)):
        img_path = os.path.join(data_dir, obj_name, f"{obj_name}.{i}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image at {img_path}")
        images.append(img)
        
    images = np.stack(images, axis=0)  # Shape: (N, H, W)
    return lights, mask, images

def compute_normals(lights, mask, images):
    N_lights, H, W = images.shape
    
    # Flatten the images and apply mask
    # I shape: (N_lights, num_valid_pixels)
    valid_pixels = images[:, mask].astype(np.float32)
    
    # Solve I = L * G, where G = normals * albedo
    # L shape: (N_lights, 3)
    # G shape: (3, num_valid_pixels)
    # L * G = I => G = (L^T L)^-1 L^T I = L_pinv * I
    
    L_pinv = np.linalg.pinv(lights)  # Shape: (3, N_lights)
    G = L_pinv @ valid_pixels        # Shape: (3, num_valid_pixels)
    
    # Albedo is the magnitude of G
    albedo = np.linalg.norm(G, axis=0) # Shape: (num_valid_pixels,)
    
    # Avoid division by zero
    valid_albedo_mask = albedo > 1e-6
    normals_valid = np.zeros_like(G)
    normals_valid[:, valid_albedo_mask] = G[:, valid_albedo_mask] / albedo[valid_albedo_mask]
    
    # Reconstruct the normal map and albedo map
    normal_map = np.zeros((H, W, 3), dtype=np.float32)
    albedo_map = np.zeros((H, W), dtype=np.float32)
    
    normal_map[mask] = normals_valid.T
    albedo_map[mask] = albedo
    
    return normal_map, albedo_map

def frankot_chellappa(p, q):
    H, W = p.shape
    u = np.fft.fftfreq(W) * 2 * np.pi
    v = np.fft.fftfreq(H) * 2 * np.pi
    U, V = np.meshgrid(u, v)
    
    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)
    
    numerator = -1j * U * P - 1j * V * Q
    denominator = U**2 + V**2
    denominator[0, 0] = 1
    
    Z_freq = numerator / denominator
    Z_freq[0, 0] = 0
    
    Z = np.real(np.fft.ifft2(Z_freq))
    return Z

def reconstruct_depth(normal_map, mask):
    nx = normal_map[..., 0]
    ny = normal_map[..., 1]
    nz = normal_map[..., 2]
    
    nz_safe = np.copy(nz)
    nz_safe[np.abs(nz) < 1e-6] = 1e-6
    
    p = -nx / nz_safe
    q = ny / nz_safe
    
    p[~mask] = 0
    q[~mask] = 0
    
    Z = frankot_chellappa(p, q)
    
    # Shift so minimum depth is 0 on the object
    if np.any(mask):
        Z = Z - np.min(Z[mask])
    Z[~mask] = 0
    return Z

def visualize_and_save(normal_map, albedo_map, depth_map, obj_name="buddha"):
    # Normalize albedo for display
    albedo_disp = cv2.normalize(albedo_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert normals from [-1, 1] to [0, 255] for RGB visualization
    # Normal mapping typically uses: R=X, G=Y, B=Z
    # Note: X is right, Y is up/down, Z is towards viewer.
    normal_disp = ((normal_map + 1.0) / 2.0 * 255.0).astype(np.uint8)
    
    # Since background is [-1, -1, -1] mapped to [0, 0, 0], we can apply mask to keep background black
    bg_mask = np.linalg.norm(normal_map, axis=2) < 1e-6
    normal_disp[bg_mask] = [0, 0, 0]
    
    # Save the results
    cv2.imwrite(f"{obj_name}_normal_map.png", cv2.cvtColor(normal_disp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{obj_name}_albedo_map.png", albedo_disp)
    
    depth_disp = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f"{obj_name}_depth_map.png", depth_disp)
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Albedo Map")
    plt.imshow(albedo_disp, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Surface Normal Map")
    plt.imshow(normal_disp)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Depth Map (Poisson)")
    plt.imshow(depth_map, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{obj_name}_photometric_stereo_result.png", dpi=300)
    plt.close()
    
    print(f"[+] Saved {obj_name}_normal_map.png")
    print(f"[+] Saved {obj_name}_albedo_map.png")
    print(f"[+] Saved {obj_name}_depth_map.png")
    print(f"[+] Saved {obj_name}_photometric_stereo_result.png")

def main():
    objects = ["buddha", "cat", "owl"]
    for obj in objects:
        print(f"\nProcessing {obj}...")
        try:
            lights, mask, images = load_data(obj_name=obj)
            normal_map, albedo_map = compute_normals(lights, mask, images)
            depth_map = reconstruct_depth(normal_map, mask)
            visualize_and_save(normal_map, albedo_map, depth_map, obj_name=obj)
        except Exception as e:
            print(f"Failed to process {obj}: {e}")

if __name__ == "__main__":
    main()
