import os
import shutil
from sklearn.model_selection import train_test_split

def dir_structure(data_folder, output_folder, test_size):
    # data_folder = "testing_jargon/face_test/cropped_images/"
    # output_folder = "testing_jargon/face_split"
    # test_size = 0.2
    class_folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    for class_folder in class_folders:
        class_path = os.path.join(data_folder, class_folder)
        image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
        train_folder = os.path.join(output_folder, "train", class_folder)
        test_folder = os.path.join(output_folder, "test", class_folder)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        for file in train_files:
            src_path = os.path.join(class_path, file)
            dst_path = os.path.join(train_folder, file)
            shutil.copy2(src_path, dst_path)
        for file in test_files:
            src_path = os.path.join(class_path, file)
            dst_path = os.path.join(test_folder, file)
            shutil.copy2(src_path, dst_path)
