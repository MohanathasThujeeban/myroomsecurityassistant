from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import OwnerProfile

DEFAULT_PROFILE_PATH = Path("data/owner_profile.npz")


def owner_profile_exists(path: Path = DEFAULT_PROFILE_PATH) -> bool:
    return path.exists()


def delete_owner_profile(path: Path = DEFAULT_PROFILE_PATH) -> bool:
    if not path.exists():
        return False

    path.unlink()
    return True


def save_owner_profile(profile: OwnerProfile, path: Path = DEFAULT_PROFILE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        owner_name=np.array(profile.owner_name),
        face_encodings=profile.face_encodings.astype(np.float32),
        body_signature=profile.body_signature.astype(np.float32),
        created_at=np.array(profile.created_at),
    )


def load_owner_profile(path: Path = DEFAULT_PROFILE_PATH) -> OwnerProfile:
    if not path.exists():
        raise FileNotFoundError(f"Owner profile was not found at {path}")

    data = np.load(path, allow_pickle=False)
    face_encodings = np.asarray(data["face_encodings"], dtype=np.float32)
    body_signature = np.asarray(data["body_signature"], dtype=np.float32)
    owner_name = str(data["owner_name"].item())
    created_at = str(data["created_at"].item())

    if face_encodings.ndim != 2 or face_encodings.shape[1] != 128:
        raise ValueError("Saved owner profile face encoding shape is invalid.")

    if body_signature.ndim != 1:
        raise ValueError("Saved owner profile body signature shape is invalid.")

    return OwnerProfile(
        owner_name=owner_name,
        face_encodings=face_encodings,
        body_signature=body_signature,
        created_at=created_at,
    )
