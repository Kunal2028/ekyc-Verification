import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from arcface import get_embedding, cosine_similarity

app = FastAPI(
    title="eKYC Face Verification API",
    description="Compare an ID image with a live image using ArcFace",
    version="1.0.0",
)

# ----------------------------
# CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {
        "message": "eKYC Face Verification API",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ----------------------------
# Face Matching API
# ----------------------------
@app.post("/match-faces")
async def match_faces(
    id_image: UploadFile = File(...),
    live_image: UploadFile = File(...),
    threshold: float = 0.35,
):
    allowed_types = {
        "image/jpeg",
        "image/png",
    }

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
        # ----------------------------
        # Save uploaded images
        # ----------------------------
        id_ext = os.path.splitext(id_image.filename or "")[1]
        live_ext = os.path.splitext(live_image.filename or "")[1]

        if id_ext == "":
            id_ext = ".jpg"

        if live_ext == "":
            live_ext = ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=id_ext,
        ) as temp:
            temp.write(await id_image.read())
            id_path = temp.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=live_ext,
        ) as temp:
            temp.write(await live_image.read())
            live_path = temp.name

        # ----------------------------
        # Extract embeddings
        # ----------------------------
        id_embedding = get_embedding(id_path)
        live_embedding = get_embedding(live_path)

        if id_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in ID image.",
            )

        if live_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in Live image.",
            )

        # ----------------------------
        # Compare
        # ----------------------------
        similarity = cosine_similarity(
            id_embedding,
            live_embedding,
        )

        matched = similarity >= threshold

        return JSONResponse(
            content={
                "success": True,
                "matched": matched,
                "similarity_score": round(float(similarity), 4),
                "threshold": threshold,
                "message": (
                    "Faces matched successfully."
                    if matched
                    else "Faces did not match."
                ),
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "detail": f"Face matching failed: {str(e)}",
            },
        )

    finally:
        for file_path in [id_path, live_path]:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)


# ----------------------------
# Run locally
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
