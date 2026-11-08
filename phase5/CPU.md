# CPU runtime used for the recorded experiment

The standard macOS MLX wheel requires a Metal device at import. This session exposed the Apple CPU only. On this machine, the following built the matching MLX core against Accelerate. Run these commands from a scratch directory with a dedicated Python 3.12 virtual environment active. The library replacement is local to that environment.

```sh
python -m pip install 'mlx==0.32.4' 'mlx-lm[train]==0.32.2' pytest
git clone --depth 1 --branch v0.32.4 https://github.com/ml-explore/mlx.git mlx-src
cmake -S mlx-src -B mlx-build -DMLX_BUILD_METAL=OFF -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build mlx-build -j 6
python - <<'PY'
import shutil, sys, sysconfig
from pathlib import Path
assert sys.prefix != sys.base_prefix, 'Activate the dedicated virtual environment first'
lib = Path(sysconfig.get_paths()['purelib']) / 'mlx/lib/libmlx.dylib'
shutil.copy2('mlx-build/libmlx.dylib', lib)
import mlx.core as mx
print(mx.default_device())
PY
```
