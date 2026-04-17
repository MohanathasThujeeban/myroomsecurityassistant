from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class OwnerProfile:
    owner_name: str
    face_encodings: np.ndarray
    body_signature: Optional[np.ndarray] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
