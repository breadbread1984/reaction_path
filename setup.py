#!/usr/bin/python3

from setuptools import setup

setup(
  name = "reaction_path",
  version = "1.0",
  author = "xieyi",
  author_email = "yxie.lhi@gmail.com",
  description = "a tool to prediction precursors of given material",
  keywords = "material, reaction, precursor",
  url = "https://github.com/breadbread1984/reaction_path/tree/main",
  packages = ["reaction_path"],
  install_requires = ["numpy","torch","transformers","pymatgen","gdown"],
  license = "Apache License 2.0"
)

