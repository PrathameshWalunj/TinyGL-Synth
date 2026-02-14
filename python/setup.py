from setuptools import setup, find_packages

setup(
    name="tinygl-synth",
    version="0.1.0",
    description="CPU renderer for embodied AI synthetic data",
    author="Prathamesh",
    author_email="prathameshwalunj27@gmail.com",
    url="https://github.com/PrathameshWalunj/tinygl-synth",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],
        "mujoco": ["mujoco>=3.0.0"],
        "gym": ["gymnasium>=0.28.0"],
        "dev": ["imageio", "matplotlib", "opencv-python"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
    ],
    keywords="synthetic-data robotics machine-learning renderer simulation",
)
