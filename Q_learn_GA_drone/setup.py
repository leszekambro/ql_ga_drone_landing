from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'Q_learn_GA_drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'launch'),
        glob(os.path.join('launch', '*.py*'))),
        (os.path.join('share', package_name,'worlds'),
        glob(os.path.join('worlds', '*.world*'))),
        (os.path.join('share', package_name,'worlds'),
        glob(os.path.join('worlds', '*.yaml*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ssdubuntu2004',
    maintainer_email='ssdubuntu2004@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "qlearnV6 = Q_learn_GA_drone.test_qlearnV6:main",
            "PyGadV6 = Q_learn_GA_drone.PyGadV6:main",
            "TestPyGadV6 = Q_learn_GA_drone.TestPyGadV6:main",
            "drone_qlearn_test = Q_learn_GA_drone.dron_qlearnV2:main"
        ],
    },
)
