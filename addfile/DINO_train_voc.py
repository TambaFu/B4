import subprocess
import sys

def main(coco_path):
    # Define the command as a list of arguments
    command = [
        "python", "main.py",
        '--dataset_file', "voc",
        "--output_dir", "logs/DINO/voc",
        "-c", "config/DINO/DINO_4scale_faster_vit_0_224_voc.py",
        "--coco_path", coco_path,
        "--options",
        "dn_scalar=100",
        "embed_init_tgt=TRUE",
        "dn_label_coef=1.0",
        "dn_bbox_coef=1.0",
        "use_ema=False",
        "dn_box_noise_scale=1.0"
    ]

    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")

if __name__ == "__main__":
    # Ensure the script is called with the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <coco_path>")
        sys.exit(1)

    # Pass the COCO path to the main function
    coco_path = sys.argv[1]
    main(coco_path)
