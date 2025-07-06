from pydantic import BaseModel

class MapParam(BaseModel):
    width  : int = 150
    height : int = 60
    seed   : int = 42

class ObstacleParam(BaseModel):
    probability : float = 0.005
    maxSize     : float = 10
    value       : int   = 1000

class EnvironmentParam(BaseModel):
    map: MapParam = MapParam()
    obstacle: ObstacleParam = ObstacleParam()