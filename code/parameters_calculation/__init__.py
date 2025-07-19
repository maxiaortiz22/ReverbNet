"""
Convenience imports for acoustic parameter calculation utilities.

This subpackage aggregates the most frequently used functions so they can be
imported from a single location, for example::

    from code.parameters_calculation import TAE, tr_lundeby, pink_noise

Wildcards (``*``) are used to preserve the original public symbols exactly as
defined in each module. The explicit ``__all__`` list below narrows the public
API when ``from ... import *`` is invoked by callers.
"""

from .tae import *
from .tr_lundeby import *
from .pink_noise import *
from .drr_augmentation import *
from .tr_augmentation import *

# Export all public functions and classes.
__all__ = [
    # TAE
    'TAE',
    
    # TR Lundeby
    'tr_lundeby',
    
    # Pink noise
    'pink_noise',
    
    # DRR augmentation
    'drr_aug',
    'get_DRR',
    
    # TR augmentation
    'tr_augmentation',
    'TrAugmentationError'
]