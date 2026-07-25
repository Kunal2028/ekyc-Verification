```python
import os
import tempfile

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from arcface import get_embedding, cosine_similarity
from livecheck import analyze_video


app = FastAPI(
    title="eKYC Verification API",
    description=(
        "Perform video liveness detection and compare "
        "an ID image with a live-video face using ArcFace"
    ),
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",
}

MAX_IMAGE_SIZE = 5 * 1024 * 1024
MAX_VIDEO_SIZE = 30 * 1024 * 1024


async def save_upload_file(
    upload: UploadFile,
    suffix: str,
    max_size: int,
) -> str:
    contents = await upload.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail=f"{upload.filename or 'Uploaded file'} is empty.",
        )

    if len(contents) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"{upload.filename or 'Uploaded file'} is too large.",
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_file.write(contents)
        return temp_file.name


def extract_best_face_frame(
    video_path: str,
    output_path: str,
) -> bool:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        return False

    best_frame = None
    largest_face_area = 0
    frame_number = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_number % 3 != 0:
                frame_number += 1
                continue

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )

            for x, y, width, height in faces:
                face_area = width * height

                if face_area > largest_face_area:
                    largest_face_area = face_area
                    best_frame = frame.copy()

            frame_number += 1

    finally:
        capture.release()

    if best_frame is None:
        return False

    return bool(cv2.imwrite(output_path, best_frame))


def safe_remove(file_path: str | None) -> None:
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass


@app.get("/")
def root():
    return {
        "message": "eKYC Verification API",
        "version": "2.0.0",
        "features": [
            "video_liveness_detection",
            "arcface_face_matching",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "liveness_module": "available",
        "face_matching_module": "available",
    }


@app.post("/verify-ekyc")
async def verify_ekyc(
    id_image: UploadFile = File(...),
    live_video: UploadFile = File(...),
    face_threshold: float = Query(
        default=0.35,
        ge=-1.0,
        le=1.0,
    ),
):
    id_path = None
    video_path = None
    live_frame_path = None

    try:
        if id_image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="ID image must be JPEG or PNG.",
            )

        if live_video.content_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Live video must be MP4, WebM, MOV, or AVI.",
            )

        id_extension = os.path.splitext(
            id_image.filename or ""
        )[1].lower()

        video_extension = os.path.splitext(
            live_video.filename or ""
        )[1].lower()

        if not id_extension:
            id_extension = ".jpg"

        if not video_extension:
            video_extension = ".mp4"

        id_path = await save_upload_file(
            upload=id_image,
            suffix=id_extension,
            max_size=MAX_IMAGE_SIZE,
        )

        video_path = await save_upload_file(
            upload=live_video,
            suffix=video_extension,
            max_size=MAX_VIDEO_SIZE,
        )

        liveness_result = analyze_video(video_path)

        if liveness_result.decision == "FAIL_NO_FACE":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Liveness verification failed because "
                        "enough face frames were not detected."
                    ),
                    "liveness_decision": liveness_result.decision,
                    "liveness_score": round(
                        float(liveness_result.score),
                        4,
                    ),
                    "reason": liveness_result.info.get(
                        "reason",
                        "not_enough_face_frames",
                    ),
                    "frames_used": liveness_result.info.get(
                        "frames_used",
                        0,
                    ),
                },
            )

        if liveness_result.decision != "LIVE":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Liveness verification failed. "
                        "A possible spoof attempt was detected."
                    ),
                    "liveness_decision": liveness_result.decision,
                    "liveness_score": round(
                        float(liveness_result.score),
                        4,
                    ),
                    "spoof_hint": liveness_result.info.get(
                        "spoof_hint",
                        "unknown",
                    ),
                    "components": {
                        key: round(float(value), 4)
                        for key, value in liveness_result.components.items()
                    },
                },
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as temp_frame:
            live_frame_path = temp_frame.name

        frame_extracted = extract_best_face_frame(
            video_path=video_path,
            output_path=live_frame_path,
        )

        if not frame_extracted:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Liveness passed, but a suitable face "
                    "frame could not be extracted from the video."
                ),
            )

        id_embedding = get_embedding(id_path)

        if id_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the ID image.",
            )

        live_embedding = get_embedding(live_frame_path)

        if live_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="No face could be processed from the live video.",
            )

        similarity = cosine_similarity(
            id_embedding,
            live_embedding,
        )

        matched = bool(similarity >= face_threshold)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "liveness_verified": True,
                "liveness": {
                    "decision": liveness_result.decision,
                    "score": round(
                        float(liveness_result.score),
                        4,
                    ),
                    "mode": liveness_result.info.get("mode"),
                    "spoof_hint": liveness_result.info.get(
                        "spoof_hint"
                    ),
                    "frames_used": liveness_result.info.get(
                        "frames_used"
                    ),
                    "bpm": round(
                        float(
                            liveness_result.info.get(
                                "bpm",
                                0.0,
                            )
                        ),
                        2,
                    ),
                    "blink_count": liveness_result.info.get(
                        "blink_count",
                        0,
                    ),
                    "mouth_open_count": liveness_result.info.get(
                        "mouth_open_count",
                        0,
                    ),
                    "components": {
                        key: round(float(value), 4)
                        for key, value in liveness_result.components.items()
                    },
                },
                "face_verification": {
                    "matched": matched,
                    "similarity_score": round(
                        float(similarity),
                        4,
                    ),
                    "threshold": face_threshold,
                },
                "message": (
                    "Liveness verified and faces matched."
                    if matched
                    else (
                        "Liveness verified, but the ID face "
                        "and live face did not match."
                    )
                ),
            },
        )

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"eKYC verification error: {error}")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "detail": (
                    "An internal error occurred during "
                    "eKYC verification."
                ),
            },
        )

    finally:
        safe_remove(id_path)
        safe_remove(video_path)
        safe_remove(live_frame_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
```
