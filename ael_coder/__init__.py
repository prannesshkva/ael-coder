# -*- coding: utf-8 -*-
"""
Ael-Coder: Multi-Million-Token Code Intelligence Engine powered by ISOM-R2
Copyright (c) 2026 Prannessh K. V. A. (@Prannesshkva). All Rights Reserved.
"""
__version__ = "1.0.0"
__author__ = "Prannessh K. V. A."

from ael_coder.cache import ContinuousLieManifold, ISOMR2VirtualSVDCache
from ael_coder.engine import ISOMR2Engine, enable_isom_r2

__all__ = [
    "ContinuousLieManifold",
    "ISOMR2VirtualSVDCache",
    "ISOMR2Engine",
    "enable_isom_r2",
]
