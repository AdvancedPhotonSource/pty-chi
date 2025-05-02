# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

# ruff: noqa: F401

try:
    from . import api
    from . import settings
    MOVIES_INSTALLED = True
except ImportError:
    MOVIES_INSTALLED = False