import os
from sklearn.model_selection import train_test_split
import shutil


def split_data(input_folder, output_folder, test_size=0.2, random_state=None):
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    class_files = {}

    for root, dirs, files in os.walk(input_folder):
        for class_dir in dirs:
            class_path = os.path.join(root, class_dir)

            files_list = os.listdir(class_path)

            class_files[class_dir] = [os.path.join(class_path, file_name) for file_name in files_list]

    min_images_per_class = 5000
    train_paths = []
    test_paths = []

    for class_dir, files in class_files.items():
        files = files[:min_images_per_class]
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

        # Move files to the corresponding folders in the output directory
        for src in train_files:
            dst = os.path.join(train_folder, class_dir, os.path.basename(src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            train_paths.append(dst)

        for src in test_files:
            dst = os.path.join(test_folder, class_dir, os.path.basename(src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            test_paths.append(dst)


if __name__ == "__main__":
    input_folder = r"C:\Users\CD138JR\Downloads\wifibills_5000"
    output_folder = r"C:\Users\CD138JR\OneDrive - EY\Documents\Wifi_bills_5000"
    test_size = 0.3  # 30% for testing
    random_state = 42  # Set seed for reproducibility (optional)

    split_data(input_folder, output_folder, test_size, random_state)