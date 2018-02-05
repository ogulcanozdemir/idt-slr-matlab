import os


def create_and_get_directory(path, new_dir):
    return_path = os.path.join(path, new_dir)
    if not os.path.exists(return_path):
        os.makedirs(return_path)

    return return_path