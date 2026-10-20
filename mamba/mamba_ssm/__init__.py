__version__ = "1.0.1"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
# PGRMamba uses the state-space module only. Avoid importing the unrelated
# language-model/transformers stack during detector construction.
