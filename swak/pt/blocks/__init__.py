"""Flexible and composable building blocks for constructing neural-networks.

After features are embedded and combined, it is time to extract as much
information as possible to predict the desired target. One way of doing this
systematically is to repeat layers of identical internal architecture with
residual (or skip) connections between them.

"""

from .activated import ActivatedBlock
from .activated_hidden import ActivatedHiddenBlock
from .gated import GatedBlock
from .gated_hidden import GatedHiddenBlock
from .gated_activated import GatedActivatedBlock
from .skip import SkipConnection
from .repeat import Repeat
from .identity import IdentityBlock


__all__ = [
    'ActivatedBlock',
    'ActivatedHiddenBlock',
    'GatedBlock',
    'GatedHiddenBlock',
    'GatedActivatedBlock',
    'SkipConnection',
    'Repeat',
    'IdentityBlock',
]
