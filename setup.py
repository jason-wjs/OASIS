from setuptools import setup, find_packages

setup(
  name = 'tasks',
  packages = find_packages(),
  author="Zehao Yu",
  author_email="yuzh24@m.fudan.edu.cn",
  description="OASIS",
  license="MIT",
  version="0.2.0",
  install_requires=[
    "loop_rate_limiters",
    "mujoco",
    "numpy==1.26.4",
    "scipy",
    "rich",
    "opencv-python",
    "protobuf",
    "imageio[ffmpeg]",
    "redis[hiredis]",
    "onnx",
    "onnxruntime-gpu",
    "mujoco-python-viewer",
    "aiortc",
    "av",
    "warp-lang==1.0.2",
    "logging_mp==0.1.6",
  ],
  extras_require={
    "vla": [
      "Pillow",
      "tensorboard",
      "torchvision",
      "tqdm",
      "transformers>=4.30",
    ],
  },
  python_requires='>=3.10',
)
