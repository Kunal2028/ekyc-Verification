import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from arcface import cosine_similarity, get_embedding


app = FastAPI(
    title="eKYC Verification API",
    description="API for face matching and identity verification",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "eKYC Verification API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/match-faces")
async def match_faces(
    id_image: UploadFile = File(...),
    live_image: UploadFile = File(...),
    threshold: float = 0.35,
):
    allowed_types = {"image/jpeg", "image/png"}

    if id_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="ID image must be JPEG or PNG",
        )

    if live_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Live image must be JPEG or PNG",
        )

    id_path = None
    live_path = None

    try:
        id_extension = os.path.splitext(id_image.filename or "")[1] or ".jpg"
        live_extension = os.path.splitext(live_image.filename or "")[1] or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=id_extension,
        ) as id_file:
            id_file.write(await id_image.read())
            id_path = id_file.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=live_extension,
        ) as live_file:
            live_file.write(await live_image.read())
            live_path = live_file.name

        id_embedding = get_embedding(id_path)
        live_embedding = get_embedding(live_path)

        if id_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in ID image",
            )

        if live_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in live image",
            )

        similarity = cosine_similarity(
            id_embedding,
            live_embedding,
        )

        matched = similarity >= threshold

        return {
            "matched": matched,
            "similarity_score": round(float(similarity), 4),
            "threshold": threshold,
            "message": (
                "Faces matched"
                if matched
                else "Faces did not match"
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Face matching failed: {str(error)}",
        ) from error

    finally:
        for path in [id_path, live_path]:
            if path and os.path.exists(path):
                os.remove(path)
