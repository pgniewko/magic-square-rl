
from setuptools import setup
from package_info import USERNAME, VERSION

setup(name='{}-{}'.format(USERNAME, 'gym-magic-square'),
    version=VERSION,
    description='Gym User Env - Solve Magic Square problem',
    author='Pawel Gniewek',
    author_email='gniewko.pablo@gmail.com',
    license='MIT License',
    install_requires=['gym>=0.2.3', 'numpy']
)
