from pydantic import BaseModel

class Boundary(BaseModel):
    inner: float = 0.0
    outer: float = 10.0

class MV(BaseModel):
    mean     : float = 0.0
    variance : float = 10.0

class InitialCoordinate(BaseModel):
    x: float = 10.0
    y: float = 10.0

class ExploreParam(BaseModel):
    boundary   : Boundary          = Boundary()
    mv         : MV                = MV()
    coordinate : InitialCoordinate = InitialCoordinate()
    robotNum   : int               = 10
    finishRate : float             = 0.8
    finishStep : int               = 10