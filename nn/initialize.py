# nn/nn/initialize.py
# Local application/library specific imports
from nn.commons.logs import set_highlighted_excepthook
from nn.commons.library import settings_verification

# Import nn.settings in working directory if not present
settings_verification()

# Colored excepthook
set_highlighted_excepthook()
