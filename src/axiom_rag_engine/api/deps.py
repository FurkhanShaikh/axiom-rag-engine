"""FastAPI dependencies shared by the route modules."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from axiom_rag_engine.services import AppServices


def get_services(request: Request) -> AppServices:
    """The running app's services (built in the lifespan); 503 before startup."""
    services: AppServices | None = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Engine is not ready.")
    return services


# Route parameter type: ``services: Services`` injects the running app's services.
Services = Annotated[AppServices, Depends(get_services)]
