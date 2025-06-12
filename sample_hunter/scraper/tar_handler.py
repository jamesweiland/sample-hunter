import tarfile
from pathlib import Path

# Always resolve paths from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MP3_DIR = PROJECT_ROOT / '_data' / 'train'
TAR_DIR = PROJECT_ROOT / '_data' / 'tar_shards'
TAR_DIR.mkdir(exist_ok=True)

SHARD_SIZE = 500 * 1024 * 1024  # 500MB

mp3_files = sorted(MP3_DIR.glob('*.mp3'))
shard_idx = 0
current_size = 0
tar = None

for mp3_path in mp3_files:
    file_size = mp3_path.stat().st_size
    # Start new tar if needed
    if tar is None or current_size + file_size > SHARD_SIZE:
        if tar is not None:
            tar.close()
        shard_idx += 1
        tar_name = TAR_DIR / f'train-{shard_idx:05d}.tar'
        tar = tarfile.open(tar_name, 'w')
        current_size = 0
        print(f"Started {tar_name.relative_to(PROJECT_ROOT)}")
    tar.add(mp3_path, arcname=mp3_path.name)
    current_size += file_size

# Close last tar
if tar is not None:
    tar.close()
    print("Done.")
