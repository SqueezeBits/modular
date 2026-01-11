import site
import os
import sys
from pathlib import Path


def find_installed_max():
    paths = site.getsitepackages()
    if site.getusersitepackages():
        paths.append(site.getusersitepackages())
    
    for path in paths:
        max_path = Path(path) / "max"
        if max_path.exists() and max_path.is_dir():
            return max_path

    return None


def link_nvidia_libraries(site_packages_path: Path):
    nvidia_dir = site_packages_path / "nvidia"
    if not nvidia_dir.exists():
        print("Installing nvidia-cudnn-cu12...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nvidia-cudnn-cu12"])
        except Exception as e:
            print(f"Error installing nvidia-cudnn-cu12: {e}")
            return

    env_lib_dir = Path(sys.prefix) / "lib"
    if not env_lib_dir.exists():
        return

    target_subdirs = ["cudnn", "cublas"]
    
    for subdir in target_subdirs:
        src_dir = nvidia_dir / subdir
        if not src_dir.exists():
            continue
            
        for src_file in src_dir.rglob("*.so*"):
            dst_file = env_lib_dir / src_file.name
        
            if dst_file.exists() or dst_file.is_symlink():
                if dst_file.is_symlink() and os.readlink(dst_file) == str(src_file):
                    continue
                dst_file.unlink()
                
            try:
                os.symlink(src_file, dst_file)
                print(f"  Linked {src_file.name}")
            except OSError as e:
                print(f"  Failed to link {src_file.name}: {e}")


def setup_sitecustomize(site_packages_path: Path, local_max_path: Path):
    sitecustomize_path = site_packages_path / "sitecustomize.py"
    
    content = f"""import sys
import os

# Prioritize local max development version
# We use insert(0) to ensure it comes before site-packages
sys.path.insert(0, '{local_max_path.resolve()}')
"""
    
    try:
        sitecustomize_path.write_text(content)
        print(f"Created {sitecustomize_path} to prioritize local source.")
    except Exception as e:
        print(f"Failed to create sitecustomize.py: {e}")


def main():
    installed_max = find_installed_max()
    if not installed_max:
        print("Error: Could not find 'max' installed in site-packages.")
        print("Please ensure you have installed the official nightly wheel first.")
        return
    print(f"Found installed binaries at: {installed_max}")

    repo_root = Path(__file__).parent.parent
    local_max_pkg = repo_root / "max" / "python" / "max"
    
    if not local_max_pkg.exists():
        print(f"Error: Could not find local source directory at {local_max_pkg}")
        return

    # Link shared objects, __init__.py, and generated protobuf files
    for root, _, files in os.walk(installed_max):
        root_path = Path(root)
        relative_path = root_path.relative_to(installed_max)
        local_dest_dir = local_max_pkg / relative_path

        if not local_dest_dir.exists():
            pass 

        for file in files:
            is_shared_object = file.endswith(".so") or ".so." in file
            is_init = file == "__init__.py"
            is_generated = file.endswith("_pb2.py") or file.endswith("_pb2_grpc.py")

            if is_shared_object or is_init or is_generated:
                src_file = root_path / file
                dst_file = local_dest_dir / file
                
                should_link = False
                if is_shared_object or is_generated:
                    should_link = True
                elif is_init:
                    if not dst_file.exists():
                        should_link = True
                
                if should_link:
                    local_dest_dir.mkdir(parents=True, exist_ok=True)

                    if dst_file.exists() or dst_file.is_symlink():
                        dst_file.unlink()
                    
                    print(f"Linking {relative_path / file}")
                    try:
                        os.symlink(src_file, dst_file)
                    except OSError as e:
                        print(f"Failed to link: {e}")

    # Link modular lib
    installed_modular_lib = installed_max.parent / "modular" / "lib"
    local_modular_dir = local_max_pkg.parent / "modular"
    
    if installed_modular_lib.exists():
        if not local_modular_dir.exists():
             local_modular_dir.mkdir(parents=True, exist_ok=True)
             
        local_lib_link = local_modular_dir / "lib"

        if local_lib_link.is_symlink():
            local_lib_link.unlink()

        print(f"Linking dependency: {installed_modular_lib} -> {local_lib_link}")
        try:
            os.symlink(installed_modular_lib, local_lib_link)
        except OSError as e:
            print(f"Failed to link modular lib: {e}")

    # Link NVIDIA libraries
    site_packages = installed_max.parent
    link_nvidia_libraries(site_packages)

    # Create sitecustomize.py to prioritize local source
    local_max_path = repo_root / "max" / "python"
    setup_sitecustomize(site_packages, local_max_path)


if __name__ == "__main__":
    main()
