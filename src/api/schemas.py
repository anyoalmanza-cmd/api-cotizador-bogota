from pydantic import BaseModel

class RequestInmueble(BaseModel):
    tipo: str
    area: float
    habitaciones: float
    banos: float
    barrio: str
    upz: str
