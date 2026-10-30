from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'bead_metrology'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'ultralytics',
        'opencv-python',
        'numpy'
    ],
    zip_safe=True,
    maintainer='Robotics Metrology Specialist',
    maintainer_email='robotics-metrology@example.com',
    description='ROS 2 package for live 3D concrete printing bead width metrology and quality control.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bead_metrology_node = bead_metrology.bead_metrology_node:main',
        ],
    },
)
