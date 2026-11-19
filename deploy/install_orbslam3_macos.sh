#!/usr/bin/env bash
# Install ORB-SLAM3 + its Python bindings on the Mac mini, for the global nav layer's visual SLAM
# (vision_service/vslam.py). Until this has run, the service works without it (backend "none":
# the global map runs on the Pi's odometry alone).
#
#   bash deploy/install_orbslam3_macos.sh            # into ~/.robot-vision/orbslam3
#   ORB_PREFIX=/opt/orb bash deploy/install_orbslam3_macos.sh
#
# What it does, each step stopping the script if it fails:
#   1. Homebrew dependencies (cmake, eigen, opencv, boost, boost-python3, glew, openssl)
#   2. Pangolin (ORB-SLAM3's viewer library; built from source, the viewer stays off)
#   3. ORB-SLAM3 itself, with its bundled DBoW2 and g2o
#   4. the ORB vocabulary (ORBvoc.txt), where VS_ORB_VOCAB expects it
#   5. the Python bindings (ORB_SLAM3-PythonBindings), into the service's Python
#   6. a check: `import orbslam3` and building a System with the vocabulary
#
# ORB-SLAM3 is developed on Ubuntu; macOS (Apple Silicon) builds need the C++ standard and
# -march flags adjusted, which step 3 does. If a step fails, the message says which; the
# service keeps running without visual SLAM in the meantime. ORB-SLAM3 is GPLv3.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${ORB_PREFIX:-$HOME/.robot-vision/orbslam3}"
SRC="$PREFIX/src"
PY="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
mkdir -p "$SRC"
step() { echo; echo "==> $*"; }
fail() { echo "!! $*" >&2; exit 1; }

step "1/6 Homebrew dependencies"
command -v brew >/dev/null || fail "Homebrew is needed: https://brew.sh"
brew install cmake eigen opencv boost boost-python3 glew openssl@3 pkg-config wget

step "2/6 Pangolin"
if [ ! -d "$SRC/Pangolin" ]; then
  git clone --depth 1 --branch v0.9.1 https://github.com/stevenlovegrove/Pangolin.git "$SRC/Pangolin"
fi
cmake -S "$SRC/Pangolin" -B "$SRC/Pangolin/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PANGOLIN_PYTHON=OFF \
  || fail "Pangolin: cmake failed"
cmake --build "$SRC/Pangolin/build" -j"$JOBS" --target install || fail "Pangolin: build failed"

step "3/6 ORB-SLAM3"
if [ ! -d "$SRC/ORB_SLAM3" ]; then
  git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$SRC/ORB_SLAM3"
fi
cd "$SRC/ORB_SLAM3"
# macOS / clang: C++14 -> C++17 (newer Eigen and Pangolin need it); no -march=native on arm64
sed -i '' -e 's/-march=native//g' -e 's/CMAKE_CXX_STANDARD 14/CMAKE_CXX_STANDARD 17/g' \
  -e 's/std=c++11/std=c++17/g' -e 's/std=c++14/std=c++17/g' \
  CMakeLists.txt Thirdparty/DBoW2/CMakeLists.txt Thirdparty/g2o/CMakeLists.txt Thirdparty/Sophus/CMakeLists.txt 2>/dev/null || true
for d in Thirdparty/DBoW2 Thirdparty/g2o; do
  cmake -S "$d" -B "$d/build" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4" \
    || fail "$d: cmake failed"
  cmake --build "$d/build" -j"$JOBS" || fail "$d: build failed"
done
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DOpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4" -DEigen3_DIR="$(brew --prefix eigen)/share/eigen3/cmake" \
  -DOPENSSL_ROOT_DIR="$(brew --prefix openssl@3)" || fail "ORB-SLAM3: cmake failed"
cmake --build build -j"$JOBS" --target ORB_SLAM3 || fail "ORB-SLAM3: build failed (see above; usually a clang flag)"

step "4/6 ORB vocabulary"
if [ ! -f "$PREFIX/ORBvoc.txt" ]; then
  tar -xzf Vocabulary/ORBvoc.txt.tar.gz -C "$PREFIX" || fail "could not unpack Vocabulary/ORBvoc.txt.tar.gz"
fi

step "5/6 Python bindings (into $PY)"
if [ ! -d "$SRC/ORB_SLAM3-PythonBindings" ]; then
  git clone --depth 1 https://github.com/hello-binit/ORB_SLAM3-PythonBindings.git "$SRC/ORB_SLAM3-PythonBindings"
fi
cd "$SRC/ORB_SLAM3-PythonBindings"
PYVER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DORB_SLAM3_DIR="$SRC/ORB_SLAM3" -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DPython3_EXECUTABLE="$PY" -DBoost_PYTHON_VERSION="${PYVER:0:1}.${PYVER:1}" \
  -DOpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4" || fail "bindings: cmake failed"
cmake --build build -j"$JOBS" || fail "bindings: build failed"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"
cp build/orbslam3*.so "$SITE"/ || fail "bindings: no orbslam3*.so produced"
# the module finds libORB_SLAM3 at run time
install_name_tool -add_rpath "$SRC/ORB_SLAM3/lib" "$SITE"/orbslam3*.so 2>/dev/null || true
install_name_tool -add_rpath "$PREFIX/lib" "$SITE"/orbslam3*.so 2>/dev/null || true

step "6/6 Check"
cd "$REPO"
"$PY" - <<PYEOF || fail "orbslam3 imports but will not start: see above"
import orbslam3, pathlib, tempfile
from vision_service.vslam import settings_yaml
cfg = settings_yaml(pathlib.Path(tempfile.mkdtemp()) / "cam.yaml", 640, 480, 80.0)
s = orbslam3.System("$PREFIX/ORBvoc.txt", str(cfg), orbslam3.Sensor.MONOCULAR)
s.set_use_viewer(False); s.initialize(); s.shutdown()
print("orbslam3 OK")
PYEOF

echo
echo "Done. Restart the vision service; GET /v1/nav/state should show vslam backend \"orbslam3\"."
echo "Vocabulary: $PREFIX/ORBvoc.txt (VS_ORB_VOCAB). Calibrate the camera for best results:"
echo "  fx/fy/cx/cy and k1 k2 p1 p2 in vslam.settings_yaml are only estimated from the 80 deg field of view."
