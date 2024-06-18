#!/usr/bin/python3

from setuptools import setup
from setuptools.command.install import install
from gdown import download

class PostInstallCommand(install):
  def run(self):
    install.run(self)
    download(id = '1ack7mcyHtUVMe99kRARvdDV8UhweElJ4', output = '')

s = setup(
  name = "reaction_path",
  version = "1.0",
  author = "xieyi",
  author_email = "yxie.lhi@gmail.com",
  description = "a tool to prediction precursors of given material",
  keywords = "material, reaction, precursor",
  url = "https://github.com/breadbread1984/reaction_path/tree/main",
  packages = ["reaction_path"],
  install_requires = ["numpy","torch","transformers","pymatgen","gdown"],
  cmdclass = {'install': PostInstallCommand},
  license = "Apache License 2.0"
)

s.command_obj['install'].__dir__()
