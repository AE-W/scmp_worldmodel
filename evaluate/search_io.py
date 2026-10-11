"""Atomic JSON artifacts and file provenance for experiment stages."""
import hashlib
import json
from pathlib import Path
import uuid
import platform
import subprocess
from importlib import metadata


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def software_versions():
    result = dict(python=platform.python_version(),platform=platform.platform())
    for name in ('torch','diffusers','triton','numpy','safetensors','accelerate','omegaconf',
                 'transformers','einops','opencv-python','imageio','imageio-ffmpeg',
                 'scikit-image','torchvision','Pillow'):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def hardware_versions(torch, device='cuda:0'):
    """Numerical hardware/runtime identity for the logical evaluation device."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA hardware identity requires an available evaluation GPU')
    properties = torch.cuda.get_device_properties(device)
    try:
        drivers = subprocess.check_output(
            ['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
            text=True,timeout=10,stderr=subprocess.DEVNULL)
        versions = sorted(set(line.strip() for line in drivers.splitlines() if line.strip()))
        driver = ','.join(versions) or None
    except (OSError,subprocess.SubprocessError):
        path = Path('/proc/driver/nvidia/version')
        driver = path.read_text().strip() if path.is_file() else None
    return dict(gpu_name=properties.name,
        compute_capability=[properties.major,properties.minor],
        total_memory=properties.total_memory,
        cuda_runtime=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        driver_version=driver)


def file_manifest(paths):
    files = {}
    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(path)
        for file in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if file.is_file():
                files[str(file)] = digest(file)
    return files


def runtime_sources():
    paths = []
    for root in ('models','diffusion','evaluate','sample','dataset','kernels/scmp_kernels'):
        paths.extend(sorted(Path(root).rglob('*.py')))
    paths.extend(sorted(Path('configs/base').glob('*.yaml')))
    paths.extend([Path('RUN_JOINT_SEARCH.sh'),Path('util.py')])
    return file_manifest(paths)


def prepare_video_run(root,spec):
    """Reuse a complete hash-verified run, or create a fresh attempt."""
    root = Path(root)
    token = fingerprint(spec)
    path = root/token
    pointer = root/(token+'.pointer.json')
    if pointer.is_file():
        name = json.loads(pointer.read_text()).get('directory','')
        if name.startswith(token) and Path(name).name==name:
            path = root/name
    manifest = path/'run_manifest.json'
    if manifest.is_file():
        previous = json.loads(manifest.read_text())
        artifacts = previous.get('artifacts',{})
        if (previous.get('complete') and previous.get('spec')==spec and artifacts and
                all((path/name).is_file() and digest(path/name)==sha for name,sha in artifacts.items())):
            return path,True
    if path.exists():
        path = root/(token+'-'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True,exist_ok=False)
    save(path/'run_manifest.json',dict(spec=spec,complete=False))
    return path,False


def complete_video_run(path):
    path = Path(path)
    manifest = json.loads((path/'run_manifest.json').read_text())
    artifacts = {str(p.relative_to(path)):digest(p) for p in sorted(path.rglob('*.json'))
                 if p!=path/'run_manifest.json'}
    if not artifacts:
        raise ValueError('Video run has no result artifacts')
    manifest.update(complete=True,artifacts=artifacts)
    save(path/'run_manifest.json',manifest)
    save(path.parent/(fingerprint(manifest['spec'])+'.pointer.json'),dict(directory=path.name))
