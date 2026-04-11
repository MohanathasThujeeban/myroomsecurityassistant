from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class OwnerProfile:
    owner_name: str
    face_encodings: np.ndarray
    body_signature: np.ndarray
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
