import os
import shutil
from typing import Tuple

# assert that the structure is as expected
def check_structure(path: str):
    def check_numeric_children_consecutive(path: str, require_start_zero: bool = True) -> Tuple[int, int]:
        """
        Validate that:
        * `path` exists and is a directory with at least one child.
        * Immediate children are directories named in canonical natural-number form:
            "0", "1", "2", ... (no leading zeros except "0").
        * Their integer values form a consecutive range from min to max.
        * If require_start_zero is True, the range must start at 0.

        Returns:
            (min_value, max_value)

        Raises:
            AssertionError on any violation.
        """
        if not os.path.isdir(path):
            raise AssertionError(f"{path!r} is not a directory")
        entries = os.listdir(path)
        assert entries, f"{path!r} is empty"

        nat_canonical = re.compile(r"0|[1-9][0-9]*\Z")
        nums = []
        for name in entries:
            full = os.path.join(path, name)
            assert os.path.isdir(full), f"{full!r} is not a directory"
            assert nat_canonical.fullmatch(name), (
                f"{name!r} is not a canonical natural number ('0', '1', '2', ... without leading zeros)"
            )
            nums.append(int(name))

        min_n, max_n = min(nums), max(nums)
        if require_start_zero:
            assert min_n == 0, f"Sequence must start at 0 but starts at {min_n}"
        expected = set(range(min_n, max_n + 1))
        actual = set(nums)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            msg = f"Immediate numeric directory names {sorted(entries)} do not form a consecutive range {min_n}..{max_n}"
            if missing:
                msg += f"; missing {missing}"
            if extra:
                msg += f"; unexpected {extra}"
            raise AssertionError(msg)
        return min_n, max_n
    # assert that it's a directory
    assert os.path.isdir(path), f"{path} is not a directory"

    # assert directory is not empty
    assert any(os.listdir(path)), f"{path} is empty"

    # assert only folders in the first level
    assert all(os.path.isdir(os.path.join(path, subdir)) for subdir in os.listdir(path)), f"Not all items in {path} are directories"

    # assert that all first level subdirectories are nats and ordered
    check_numeric_children_consecutive(path)

    # go into folder titled path/0 and find the set of folder names
    zero_folder = os.path.join(path, "0")
    assert os.path.isdir(zero_folder), f"{zero_folder} is not a directory"

    folder_names = {name for name in os.listdir(zero_folder) if os.path.isdir(os.path.join(zero_folder, name))}

    # assert that set of folder_names is the same across all number directories
    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path) and subdir != "0":
            subdir_folder_names = {name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))}
            assert subdir_folder_names == folder_names, f"Folder names in {subdir_path} do not match those in {zero_folder}"
    
    # assert that all subfolders have the same file names e.g. path/0/subfolder/[1..8].txt matches path/1/
    file_set = None
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for subdir in os.listdir(folder_path):
            assert os.path.isdir(os.path.join(folder_path, subdir)), f"{subdir} is not a directory"
            file_names = {name for name in os.listdir(os.path.join(folder_path, subdir)) if os.path.isfile(os.path.join(folder_path, subdir, name))}
            if file_set is None:
                file_set = file_names
            else:
                assert file_set == file_names, f"File names in {subdir} are not consistent with other subdirectories"

def find_intersection_of_number_file_subfolders(path: str) -> set:
    folder_names = {name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))}
    print(folder_names)
    intersection_set = None

    for folder in folder_names:
        folder_path = os.path.join(path, folder)
        assert os.path.isdir(folder_path), f"{folder_path} is not a directory"
        subfolder_names = {name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))}
        if intersection_set is None:
            intersection_set = subfolder_names
        else:
            intersection_set &= subfolder_names

    return intersection_set

def find_non_intersection_set_folders(path:str, delete: bool = False):
    intersection_set = find_intersection_of_number_file_subfolders(path)
    print(f"Intersection set: {intersection_set}")

    folder_names = {name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))}
    for folder in folder_names:
        folder_path = os.path.join(path, folder)
        assert os.path.isdir(folder_path), f"{folder_path} is not a directory"
        subfolder_names = {name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))}
        non_intersection = subfolder_names - intersection_set
        for subfolder in non_intersection:
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                print(f"{subfolder_path}")
                if delete:
                    shutil.rmtree(subfolder_path)