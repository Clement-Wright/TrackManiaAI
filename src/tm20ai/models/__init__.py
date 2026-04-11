"""Model definitions for Trackmania learning experiments."""

from .full_actor_critic import FullObservationActor, FullObservationCritic
from .lidar_actor_critic import LidarActor, LidarCritic

__all__ = [
    "FullObservationActor",
    "FullObservationCritic",
    "LidarActor",
    "LidarCritic",
]
