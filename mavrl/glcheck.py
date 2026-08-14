#!/usr/bin/env python3
"""Does this box render on the GPU, headless? Answer in ten seconds.

    python -m mavrl.glcheck

A training run that renders in software still *works*, which is the trap: it
finishes, it logs, it just takes a week. So this does not stop at "did a frame
come back" -- it names the renderer string and measures throughput, because the
only symptom distinguishing llvmpipe from an RTX card is the clock.

Four tiers, each one useless to reach if the one before it failed:

  1. environment   MUJOCO_GL, the driver, the EGL vendor ICDs
  2. context       can a GL context be created at all, headless
  3. renderer      *which* GL implementation answered -- GPU or software
  4. course        the real 512x512 offscreen render, at the real settings

Exit code is 0 only when tier 4 renders on hardware. `--allow-software` lowers
that bar to "rendered at all", for a box where you already know what you have.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Renderer strings that mean "the CPU is doing this". Substring match, lowered.
#: Deliberately not matching "gallium" or "mesa" on their own -- Mesa is also the
#: real driver for AMD and Intel hardware, and flagging those would be wrong.
SOFTWARE_MARKERS = ("llvmpipe", "softpipe", "swrast", "software rasterizer",
                    "mesa offscreen")

#: A model with nothing in it but a lit box. If this cannot render, nothing can,
#: and the failure is GL's rather than anything the course XML asked for.
TRIVIAL_XML = """
<mujoco>
  <visual><global offwidth="64" offheight="64"/><quality offsamples="0"/></visual>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="box" size=".5 .5 .5" rgba="1 .3 .1 1"/>
    <camera name="cam" pos="2 2 2" xyaxes="-1 1 0 -.5 -.5 1"/>
  </worldbody>
</mujoco>
"""


def _say(ok: bool, label: str, detail: str = "") -> bool:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ::  {detail}" if detail else ""), flush=True)
    return ok


def explain_import_failure(exc: Exception) -> None:
    """Turn `import mujoco` blowing up under MUJOCO_GL=egl into a diagnosis.

    The signature failure is an AttributeError on None: PyOpenGL could not load
    libEGL, so it hands MuJoCo `None` and MuJoCo calls a method on it. The
    traceback names `eglQueryString` and nothing about the actual cause, which
    is that the GL libraries are not installed -- routinely true on a GPU box,
    because NVIDIA's own datacenter instructions install the driver with
    `--no-opengl-files`, giving you CUDA and no EGL.
    """
    print("\n  why:")
    if not sys.platform.startswith("linux"):
        print(f"    {exc!r}")
        return

    import ctypes.util
    found = ctypes.util.find_library("EGL")
    print(f"    ctypes.util.find_library('EGL') = {found!r}")

    try:
        import OpenGL
        print(f"    PyOpenGL                        = {OpenGL.__version__}")
    except ImportError:
        print("    PyOpenGL                        = NOT INSTALLED"
              "   ->  pip install PyOpenGL")

    libs = {}
    for soname in ("libEGL.so.1", "libGLX.so.0", "libOpenGL.so.0",
                   "libEGL_nvidia.so.0"):
        try:
            ctypes.CDLL(soname)
            libs[soname] = "loadable"
        except OSError:
            libs[soname] = "MISSING"
    for soname, state in libs.items():
        print(f"    {soname:<22} {state}")

    if libs["libEGL.so.1"] == "MISSING":
        print("\n    libEGL is not on this machine. Either:")
        print("      sudo apt-get install -y libegl1 libglvnd0 libgles2 libglx0")
        print("    or, with no root / a driver installed --no-opengl-files,")
        print("    run in the container: docker build -t mavrl -f docker/Dockerfile .")
    elif libs["libEGL_nvidia.so.0"] == "MISSING":
        print("\n    libEGL loads but NVIDIA's does not: the driver was "
              "installed without\n    its GL components. Reinstall it without "
              "--no-opengl-files, or use the container.")


# --------------------------------------------------------------------------
# tier 1 -- environment
# --------------------------------------------------------------------------

def tier_environment() -> bool:
    """Report the environment; returns whether an NVIDIA GPU is present."""
    print("\n1. environment")
    print(f"  MUJOCO_GL      = {os.environ.get('MUJOCO_GL')!r}")
    print(f"  DISPLAY        = {os.environ.get('DISPLAY')!r}")
    print(f"  python         = {sys.version.split()[0]}  on  {sys.platform}")

    try:
        import mujoco
        print(f"  mujoco         = {mujoco.__version__}")
    except Exception as exc:
        _say(False, "import mujoco", repr(exc))
        explain_import_failure(exc)
        raise SystemExit(2)

    smi = shutil.which("nvidia-smi")
    have_gpu = False
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20).stdout.strip()
            for line in out.splitlines():
                print(f"  gpu            = {line.strip()}")
            have_gpu = bool(out)
        except Exception as exc:
            _say(False, "nvidia-smi ran but did not answer", repr(exc))
    else:
        _say(False, "no nvidia-smi on PATH",
             "in a container this usually means it was started without "
             "--gpus all / the nvidia runtime")

    # The ICD is how libglvnd finds NVIDIA's EGL. A CUDA container that has the
    # driver but not this file lands on Mesa and renders in software -- with no
    # error anywhere, which is exactly the failure worth catching here.
    if sys.platform.startswith("linux"):
        icds = sorted(glob.glob("/usr/share/glvnd/egl_vendor.d/*.json")
                      + glob.glob("/etc/glvnd/egl_vendor.d/*.json"))
        _say(any("nvidia" in p.lower() for p in icds),
             "NVIDIA EGL vendor ICD present",
             ", ".join(os.path.basename(p) for p in icds) or "none found")
    return have_gpu


def tier_egl_devices() -> None:
    """List the EGL devices libEGL can see, without going through mujoco.

    Worth doing separately: if this shows zero devices, the problem is the
    container or the driver install, and nothing about the course XML or the
    render resolution is going to matter.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        egl = ctypes.CDLL("libEGL.so.1")
        get_proc = egl.eglGetProcAddress
        get_proc.restype = ctypes.c_void_p
        get_proc.argtypes = [ctypes.c_char_p]
        addr = get_proc(b"eglQueryDevicesEXT")
        if not addr:
            _say(False, "eglQueryDevicesEXT unavailable",
                 "libEGL is too old or is not libglvnd")
            return
        proto = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_void_p),
                                 ctypes.POINTER(ctypes.c_int))
        query = proto(addr)
        n = ctypes.c_int(0)
        devs = (ctypes.c_void_p * 16)()
        query(16, devs, ctypes.byref(n))
        _say(n.value > 0, f"libEGL sees {n.value} device(s)")
    except OSError as exc:
        _say(False, "libEGL.so.1 not loadable", repr(exc))


# --------------------------------------------------------------------------
# tier 2 and 3 -- a context, and whose context it is
# --------------------------------------------------------------------------

def gl_strings() -> dict:
    """GL_VENDOR / GL_RENDERER / GL_VERSION, for whatever context is current.

    Must be called with a context already made current. glvnd splits the old
    libGL into several sonames and which one carries glGetString depends on how
    the image was built, so all of them get a try before giving up.
    """
    libs = (["opengl32.dll"] if sys.platform == "win32"
            else ["libOpenGL.so.0", "libGL.so.1", "libGLESv2.so.2"])
    for name in libs:
        try:
            lib = ctypes.CDLL(name)
            fn = lib.glGetString
        except (OSError, AttributeError):
            continue
        fn.restype = ctypes.c_char_p
        fn.argtypes = [ctypes.c_uint]
        try:
            out = {k: (fn(v) or b"").decode(errors="replace")
                   for k, v in (("vendor", 0x1F00), ("renderer", 0x1F01),
                                ("version", 0x1F02))}
        except Exception:
            continue
        if out["renderer"]:
            return out
    return {}


def is_software(renderer: str) -> bool:
    low = renderer.lower()
    return any(m in low for m in SOFTWARE_MARKERS)


def tier_context(have_gpu: bool = False) -> dict:
    """Create a context, render the trivial model, name the implementation."""
    import mujoco
    print("\n2. context + 3. renderer")

    try:
        model = mujoco.MjModel.from_xml_string(TRIVIAL_XML)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=64, width=64)
    except Exception as exc:
        _say(False, "create a 64x64 offscreen context", repr(exc))
        raise SystemExit(3)
    _say(True, "create a 64x64 offscreen context")

    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="cam")
    pix = renderer.render()
    _say(pix.shape == (64, 64, 3), "render a frame", str(pix.shape))
    # An all-black frame is the classic silent failure: the context exists, the
    # call returns, and nothing was ever drawn.
    _say(int(pix.max()) > 0, "frame is not uniformly black",
         f"max={int(pix.max())} mean={pix.mean():.1f}")

    info = gl_strings()
    if not info:
        _say(False, "read GL_RENDERER", "could not resolve glGetString")
    else:
        print(f"  GL_VENDOR      = {info['vendor']}")
        print(f"  GL_RENDERER    = {info['renderer']}")
        print(f"  GL_VERSION     = {info['version']}")
        _say(not is_software(info["renderer"]),
             "rendering on hardware",
             "software rasterizer -- this will be ~50x too slow"
             if is_software(info["renderer"]) else info["renderer"])
        # A box with a discrete card can still hand GL to the integrated one --
        # everything below passes, nvidia-smi looks healthy, and the frames come
        # off the wrong chip. Only worth saying when there is an NVIDIA GPU to
        # have missed.
        if have_gpu and "nvidia" not in info["renderer"].lower():
            info["wrong_gpu"] = "1"
            _say(False, "GL is on the NVIDIA GPU",
                 f"nvidia-smi sees a card but GL answered {info['renderer']!r}"
                 " -- on Linux/EGL set MUJOCO_GL=egl; a laptop under GLFW is "
                 "picking the iGPU")
    renderer.close()
    return info


# --------------------------------------------------------------------------
# tier 4 -- the render this project actually performs
# --------------------------------------------------------------------------

def tier_course(steps: int) -> float:
    """Step the real env, and time it.

    Tier 2 proves GL works. This proves it works at 512x512 with the arena
    loaded, which is a different question: the offscreen framebuffer has to be
    resized past MuJoCo's 640x480 default, and multisampling has to be off or
    NVIDIA's surfaceless contexts refuse the combination outright
    (GL_FRAMEBUFFER_UNSUPPORTED, 0x8CDD).

    Timing the whole `env.step` rather than a bare `renderer.render()` is
    deliberate -- it is the number that decides how long a run takes, and it
    includes the 24 physics ticks and the 512->128 downsample that a raw render
    loop would leave out.
    """
    from mavrl import config as C
    from mavrl.course_aviary import CourseAviary
    from mavrl.course_world import sample_layout
    print("\n4. the course render")

    layout = sample_layout(np.random.default_rng(0), 3)
    try:
        env = CourseAviary(layout=layout, seed=0)
    except Exception as exc:
        _say(False, "build CourseAviary", repr(exc))
        return 0.0
    _say(True, f"build CourseAviary at {C.RENDER_RES}x{C.RENDER_RES}")

    try:
        obs, _ = env.reset()
        img = obs["image"]
        _say(img.shape == (C.IMG_RES, C.IMG_RES, C.IMG_CHANNELS),
             "observation image shape", str(img.shape))
        _say(int(img[..., :3].max()) > 0, "colour channels have content",
             f"mean={img[..., :3].mean():.1f}")
        _say(int(img[..., 3].max()) > 0, "depth channel has content",
             f"mean={img[..., 3].mean():.1f}")

        action = np.zeros(C.N_ACTIONS, dtype=np.float32)
        env.step(action)                    # first step pays for warm-up
        t0 = time.perf_counter()
        for _ in range(steps):
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                env.reset()
        fps = steps / (time.perf_counter() - t0)
    except Exception as exc:
        _say(False, "step the env", repr(exc))
        return 0.0
    finally:
        env.close()

    print(f"  throughput     = {fps:.0f} policy steps/s, single env")
    # One policy step renders one frame. Below ~50 steps/s per process the
    # renderer, not the network, is what the run spends its day waiting on --
    # and a software rasterizer lands an order of magnitude under that.
    _say(fps > 50, "fast enough to train on",
         f"{fps:.0f} steps/s -- software-rasterizer territory"
         if fps <= 50 else "")
    return fps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=60,
                   help="env steps to time the course render over")
    p.add_argument("--allow-software", action="store_true",
                   help="exit 0 even on a software rasterizer")
    p.add_argument("--skip-course", action="store_true",
                   help="stop after tier 3; useful before the assets exist")
    args = p.parse_args(argv)

    have_gpu = tier_environment()
    tier_egl_devices()
    info = tier_context(have_gpu)
    fps = 0.0 if args.skip_course else tier_course(args.steps)

    soft = is_software(info.get("renderer", "")) or (
        not args.skip_course and 0 < fps <= 50)
    good = (args.skip_course or fps > 0) and (args.allow_software or not soft)

    doc = "see mavrl/README.md 'Headless GPU rendering'."
    if not info and not fps:
        verdict = f"HEADLESS RENDERING IS BROKEN -- {doc}"
    elif soft:
        verdict = f"RENDERS, BUT IN SOFTWARE -- {doc}"
    elif info.get("wrong_gpu"):
        verdict = f"RENDERS ON THE WRONG GPU -- {doc}"
    else:
        verdict = "RENDERING ON THE GPU -- go train."
    print("\n" + verdict)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
