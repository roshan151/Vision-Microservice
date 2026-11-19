#!/usr/bin/env bash
# Install ORB-SLAM3 + its Python bindings on the Mac mini, for the global nav layer's visual SLAM
# (vision_service/vslam.py). Until this has run, the service works without it (backend "none":
# the global map runs on the Pi's odometry alone).
#
#   bash deploy/install_orbslam3_macos.sh            # into ~/.robot-vision/orbslam3
#   ORB_PREFIX=/opt/orb bash deploy/install_orbslam3_macos.sh
#
# What it does, each step stopping the script if it fails:
#   1. Homebrew dependencies (cmake, eigen, opencv, boost, glew, openssl)
#   2. Pangolin (ORB-SLAM3's viewer library; built from source, the viewer stays off)
#   3. ORB-SLAM3 itself (hello-binit's fork, the one the bindings are written against), with its
#      bundled DBoW2 and g2o, laid out under the prefix the way the bindings look for it
#   4. the ORB vocabulary (ORBvoc.txt), where VS_ORB_VOCAB expects it
#   5. Boost.Python built for the service's own Python (Homebrew's boost-python3 is built for
#      Homebrew's newest Python, 3.15 as of Nov 2026; the service runs Anaconda's 3.12)
#   6. the Python bindings (ORB_SLAM3-PythonBindings, written for Python 3.10 on Ubuntu: patched for
#      this Python), into the service's Python as orbslam3.so
#   7. a check: `import orbslam3` and building a System with the vocabulary
#
# Nothing to clone by hand: everything is fetched into $ORB_PREFIX/src. Re-running skips what is there.
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
PYBASE="$("$PY" -c 'import sys, os; print(os.path.join(sys.base_prefix, "bin", "python3"))')"   # the venv's interpreter
PYDOT="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"         # 3.12
PYVER="${PYDOT/./}"                                                                                    # 312
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
# ORB-SLAM3, DBoW2, g2o and the bindings ask for CMake 2.8/3.x behaviour, which CMake 4 (Homebrew) refuses
# outright; this lets every configure below go ahead with 3.5 rules.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
mkdir -p "$SRC"
step() { echo; echo "==> $*"; }
fail() { echo "!! $*" >&2; exit 1; }

step "1/7 Homebrew dependencies"
command -v brew >/dev/null || fail "Homebrew is needed: https://brew.sh"
# OpenCV 4 and Eigen 3, not Homebrew's default 5s: ORB-SLAM3, g2o and Sophus are written for those
brew install cmake eigen@3 opencv@4 boost glew openssl@3 pkg-config wget
OPENCV_DIR="$(brew --prefix opencv@4)/lib/cmake/opencv4"
EIGEN_DIR="$(brew --prefix eigen@3)/share/eigen3/cmake"
# Apple's clang does not look in Homebrew's folders by itself (boost headers: DBoW2 includes them directly).
# These come after any -I, so OpenCV 4's own include path still wins over Homebrew's OpenCV 5 headers.
export CPATH="$(brew --prefix)/include:$(brew --prefix openssl@3)/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$(brew --prefix)/lib:$(brew --prefix openssl@3)/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# This code predates C++17 (needed by Pangolin's headers): let it keep what C++17 dropped from the library
# (binary_function, random_shuffle, auto_ptr) and the 'register' keyword, and keep deprecation noise down.
COMPAT="-Wno-register -Wno-deprecated-declarations -D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES \
-D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION -D_LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE \
-D_LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR -D_LIBCPP_ENABLE_CXX17_REMOVED_BINDERS"
resign() { codesign -f -s - "$1" >/dev/null 2>&1 || true; }   # Apple Silicon refuses to load a binary edited unsigned

step "2/7 Pangolin"
# Video input drivers off: ORB-SLAM3 only uses Pangolin's (disabled) viewer, and Pangolin 0.9.1's FFmpeg
# driver calls avcodec_close, which FFmpeg 7 (Homebrew) removed.
if [ ! -d "$SRC/Pangolin" ]; then
  git clone --depth 1 --branch v0.9.1 https://github.com/stevenlovegrove/Pangolin.git "$SRC/Pangolin"
fi
cmake -S "$SRC/Pangolin" -B "$SRC/Pangolin/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PANGOLIN_PYTHON=OFF \
  -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBDC1394=OFF -DBUILD_PANGOLIN_REALSENSE=OFF \
  -DBUILD_PANGOLIN_REALSENSE2=OFF -DBUILD_PANGOLIN_OPENNI=OFF -DBUILD_PANGOLIN_OPENNI2=OFF -DBUILD_PANGOLIN_LIBUVC=OFF \
  -DEigen3_DIR="$EIGEN_DIR" \
  || fail "Pangolin: cmake failed"
cmake --build "$SRC/Pangolin/build" -j"$JOBS" --target install || fail "Pangolin: build failed"

step "3/7 ORB-SLAM3"
if [ ! -d "$SRC/ORB_SLAM3" ]; then
  git clone --depth 1 https://github.com/hello-binit/ORB_SLAM3.git "$SRC/ORB_SLAM3"
fi
cd "$SRC/ORB_SLAM3"
# macOS / clang: C++14 -> C++17 (newer Eigen and Pangolin need it); no -march=native on arm64
sed -i '' -e 's/-march=native//g' -e 's/CMAKE_CXX_STANDARD 14/CMAKE_CXX_STANDARD 17/g' \
  -e 's/std=c++11/std=c++17/g' -e 's/std=c++14/std=c++17/g' \
  CMakeLists.txt Thirdparty/DBoW2/CMakeLists.txt Thirdparty/g2o/CMakeLists.txt Thirdparty/Sophus/CMakeLists.txt 2>/dev/null || true
# GCC-only header (Linux): the standard one does the same
grep -rl 'stdint-gcc.h' src include Thirdparty 2>/dev/null | xargs sed -i '' 's/<stdint-gcc.h>/<stdint.h>/' || true
# g2o: TR1 hash maps (pre-C++11 GCC) are plain std:: now
grep -rlE 'tr1/|tr1::' Thirdparty/g2o 2>/dev/null | xargs sed -i '' \
  -e 's#<tr1/unordered_map>#<unordered_map>#g' -e 's#<tr1/memory>#<memory>#g' \
  -e 's/std::tr1::/std::/g' -e 's/tr1::/std::/g' || true
# ORB-SLAM3 links its Thirdparty libraries by their Linux names (.so); macOS builds .dylib
sed -i '' -e 's#lib/libDBoW2\.so#lib/libDBoW2${CMAKE_SHARED_LIBRARY_SUFFIX}#g' \
  -e 's#lib/libg2o\.so#lib/libg2o${CMAKE_SHARED_LIBRARY_SUFFIX}#g' CMakeLists.txt
# a map whose allocator names the wrong pair type: libc++ rejects it (the well-known ORB-SLAM3 fix)
sed -i '' 's/std::pair<const KeyFrame\*, *g2o::Sim3>/std::pair<KeyFrame* const, g2o::Sim3>/g' include/LoopClosing.h
# They ask for OpenCV "3.0" / "4.4"; OpenCV only counts the same major version as a match, so ask for any
sed -i '' -E 's/find_package\(OpenCV [0-9.]+/find_package(OpenCV/' CMakeLists.txt Thirdparty/DBoW2/CMakeLists.txt
for d in Thirdparty/DBoW2 Thirdparty/g2o; do
  cmake -S "$d" -B "$d/build" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$OPENCV_DIR" -DEigen3_DIR="$EIGEN_DIR" \
    -DCMAKE_CXX_FLAGS="$COMPAT" \
    || fail "$d: cmake failed"
  cmake --build "$d/build" -j"$JOBS" || fail "$d: build failed"
done
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_CXX_FLAGS="$COMPAT" \
  -DOpenCV_DIR="$OPENCV_DIR" -DEigen3_DIR="$EIGEN_DIR" \
  -DOPENSSL_ROOT_DIR="$(brew --prefix openssl@3)" || fail "ORB-SLAM3: cmake failed"
cmake --build build -j"$JOBS" --target ORB_SLAM3 || fail "ORB-SLAM3: build failed (see above; usually a clang flag)"
# The bindings look for <prefix>/include/ORB_SLAM3/System.h and <prefix>/lib/libORB_SLAM3: install if the
# fork can, else link the build into that shape.
cmake --install build --prefix "$PREFIX" >/dev/null 2>&1 || true
if [ ! -f "$PREFIX/include/ORB_SLAM3/System.h" ]; then
  mkdir -p "$PREFIX/include" "$PREFIX/lib"
  ln -sfn "$SRC/ORB_SLAM3/include" "$PREFIX/include/ORB_SLAM3"
  for lib in lib/libORB_SLAM3.dylib Thirdparty/DBoW2/lib/libDBoW2.dylib Thirdparty/g2o/lib/libg2o.dylib; do
    [ -f "$lib" ] && ln -sf "$SRC/ORB_SLAM3/$lib" "$PREFIX/lib/"
  done
fi
[ -f "$PREFIX/include/ORB_SLAM3/System.h" ] || fail "ORB-SLAM3: built, but include/System.h is not where expected"

step "4/7 ORB vocabulary"
if [ ! -f "$PREFIX/ORBvoc.txt" ]; then
  tar -xzf Vocabulary/ORBvoc.txt.tar.gz -C "$PREFIX" || fail "could not unpack Vocabulary/ORBvoc.txt.tar.gz"
fi

step "5/7 Boost.Python for Python $PYDOT"
BV="$(brew list --versions boost | awk '{print $2}' | sed 's/_.*//')"     # the same Boost as the headers ORB-SLAM3 used
if ! ls "$PREFIX"/lib/libboost_python"$PYVER".* >/dev/null 2>&1; then
  BDIR="$SRC/boost_${BV//./_}"
  if [ ! -d "$BDIR" ]; then
    curl -fL "https://archives.boost.io/release/$BV/source/boost_${BV//./_}.tar.gz" | tar -xz -C "$SRC" \
      || fail "Boost $BV source: download failed"
  fi
  cd "$BDIR"
  # Name the interpreter, its headers and its library folder outright: b2 otherwise skips Boost.Python with
  # only a note ("couldn't find Python") and still succeeds. Extension-style linking: Python comes from the
  # interpreter at run time (Anaconda's libpython is a static .a; it must not be copied in).
  PYINC_B="$("$PYBASE" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
  PYLIBDIR_B="$("$PYBASE" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
  echo "using python : $PYDOT : \"$PYBASE\" : \"$PYINC_B\" : \"$PYLIBDIR_B\" ;" > "$SRC/user-config.jam"
  ./bootstrap.sh --with-libraries=python --prefix="$PREFIX" || fail "Boost.Python: bootstrap failed"
  # --with-python: Boost 1.92's install otherwise visits every library and aborts on histogram's
  # ("No best alternative ... install-libraries-shared") before Boost.Python is built. C++17 as the rest.
  ./b2 -j"$JOBS" --user-config="$SRC/user-config.jam" --debug-configuration --with-python python="$PYDOT" \
    cxxstd=17 link=shared variant=release linkflags="-undefined dynamic_lookup" install > "$SRC/b2.log" 2>&1 \
    || { grep -iE "error|python" "$SRC/b2.log" | tail -20; fail "Boost.Python: build failed (full log: $SRC/b2.log)"; }
fi
if ! ls "$PREFIX"/lib/libboost_python"$PYVER".dylib >/dev/null 2>&1; then
  grep -iE "python|skipping" "$SRC/b2.log" 2>/dev/null | tail -20
  fail "Boost.Python: b2 finished but built no libboost_python$PYVER (full log: $SRC/b2.log)"
fi
for f in "$PREFIX"/lib/libboost_python"$PYVER"*.dylib; do   # b2 leaves a bare file name as the install name
  [ -L "$f" ] && continue
  install_name_tool -id "$f" "$f" && resign "$f"
done

step "6/7 Python bindings (into $PY)"
if [ ! -d "$SRC/ORB_SLAM3-PythonBindings" ]; then
  git clone --depth 1 https://github.com/hello-binit/ORB_SLAM3-PythonBindings.git "$SRC/ORB_SLAM3-PythonBindings"
fi
cd "$SRC/ORB_SLAM3-PythonBindings"
# written for Python 3.10: ask for this one instead
sed -i '' -e "s/PythonLibs 3.10/PythonLibs $PYDOT/" -e "s/COMPONENTS python310/COMPONENTS python$PYVER/" \
  -e 's/\${PYTHON_LIBRARIES}//g' CMakeLists.txt     # an extension module takes Python from the interpreter
PYINC="$("$PYBASE" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PYLIB="$("$PYBASE" -c 'import sysconfig, os; print(os.path.join(sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY")))')"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DORB_SLAM3_DIR="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DBOOST_ROOT="$PREFIX" -DBoost_NO_SYSTEM_PATHS=ON -DPYTHON_EXECUTABLE="$PY" \
  -DCMAKE_SHARED_LINKER_FLAGS="-undefined dynamic_lookup" \
  -DPYTHON_INCLUDE_DIR="$PYINC" -DPYTHON_LIBRARY="$PYLIB" \
  -DCMAKE_CXX_FLAGS="-std=c++17 $COMPAT -I$SRC/ORB_SLAM3 -I$SRC/ORB_SLAM3/include -I$SRC/ORB_SLAM3/include/CameraModels -I$SRC/ORB_SLAM3/Thirdparty/Sophus" \
  -DOpenCV_DIR="$OPENCV_DIR" -DEigen3_DIR="$EIGEN_DIR" \
  || fail "bindings: cmake failed"
cmake --build build -j"$JOBS" || fail "bindings: build failed"
MOD="$(find lib build -maxdepth 2 \( -name 'orbslam3.so' -o -name 'orbslam3*.dylib' -o -name 'orbslam3*.so' \) 2>/dev/null | head -1)"
[ -n "$MOD" ] || fail "bindings: built, but no orbslam3 module found under lib/ or build/"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"
cp "$MOD" "$SITE/orbslam3.so" || fail "bindings: could not copy into $SITE"   # Python imports .so, even on macOS
# where it finds libORB_SLAM3, DBoW2, g2o and boost_python at run time
for rp in "$PREFIX/lib" "$SRC/ORB_SLAM3/lib" "$SRC/ORB_SLAM3/Thirdparty/DBoW2/lib" "$SRC/ORB_SLAM3/Thirdparty/g2o/lib"; do
  install_name_tool -add_rpath "$rp" "$SITE/orbslam3.so" 2>/dev/null || true
done
resign "$SITE/orbslam3.so"

step "7/7 Check"
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
