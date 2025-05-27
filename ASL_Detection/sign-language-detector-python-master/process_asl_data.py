import os
import shutil

# Define directories
TRAIN_DATA_DIR = '../full_asl_data/asl_alphabet_train/asl_alphabet_train'
TEST_DATA_DIR = '../full_asl_data/asl_alphabet_test/asl_alphabet_test'
OUTPUT_DIR = './data'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Function to copy a subset of images from each class
def process_training_data(sample_size=100):
    # Get all classes (directories) from the training data
    classes = os.listdir(TRAIN_DATA_DIR)
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Create a mapping from class name to numeric index
    class_to_index = {class_name: i for i, class_name in enumerate(sorted(classes))}
    
    # Save class mapping to a file for later reference
    with open(os.path.join(OUTPUT_DIR, 'class_mapping.txt'), 'w') as f:
        for class_name, index in class_to_index.items():
            f.write(f"{index}: {class_name}\n")
    
    # Process each class
    for class_name in classes:
        src_dir = os.path.join(TRAIN_DATA_DIR, class_name)
        dst_dir = os.path.join(OUTPUT_DIR, str(class_to_index[class_name]))
        
        # Create destination directory
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        # Get all image files
        img_files = [f for f in os.listdir(src_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Select a subset (sample_size) of images or all if less than sample_size
        selected_files = img_files[:min(sample_size, len(img_files))]
        
        # Copy selected images
        for i, img_file in enumerate(selected_files):
            src_path = os.path.join(src_dir, img_file)
            dst_path = os.path.join(dst_dir, f"{i}.jpg")
            shutil.copy(src_path, dst_path)
        
        print(f"Processed class {class_name} -> {class_to_index[class_name]}: Copied {len(selected_files)} images")

# Main execution
if __name__ == "__main__":
    print("Processing ASL training data...")
    process_training_data(sample_size = 1000)  # Adjust sample size as needed
    print("ASL data processing complete!") 