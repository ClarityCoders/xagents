from setuptools import find_packages, setup

install_requires = [dep.strip() for dep in open('requirements.txt')]

setup(
    name='xagents',
    version='1.0',
    packages=find_packages(),
    url='https://github.com/schissmantics/xagents',
    license='MIT',
    author='schissmantics',
    author_email='schissmantics@outlook.com',
    description='Implementations of deep reinforcement learning algorithms in tensorflow 2.5',
    include_package_data=True,
    setup_requires=['numpy==1.19.5'],
    install_requires=install_requires,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'xagents=xagents.cli:execute',
        ],
    },
)
