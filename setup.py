import setuptools
from numpy.distutils.core import Extension
from numpy.distutils.core import setup

setup(
   name="mlmd",
   version="0.1",
   author="Arturo Hernandez",
   author_email="my@email.com",
   description="Example package to demonstrate wheel issue",
   packages=['mlmd', 'mlmd.machine_learning', 'mlmd.tools', 'mlmd.MD_suit'],
   ext_modules=[Extension('mlmd.MD_suit.MD_suit',
                          ['mlmd/MD_suit/MD_suit.f90'])],
   scripts=['scripts/mlmd_']
)
