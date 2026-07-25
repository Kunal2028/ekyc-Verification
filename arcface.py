```python
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm


_face_app = None


def get_face_app(
    det_size=(640, 640),
    providers=None,
    ctx_id=0,
):
    global _face_app

    if _face_app is None:
        if providers is None:
            providers = ["CPUExecutionProvider"]

        _face_app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )

        _face_app.prepare(
            ctx_id=ctx_id,
            det_size=det_size,
        )

    return _face_app


def read_image_any(input_data):
    if isinstance(input_data, str):
        try:
            data = np.fromfile(
                input_data,
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                data,
                cv2.IMREAD_COLOR,
            )

        except Exception:
            image = cv2.imread(input_data)

        if image is None:
            raise ValueError(
                f"Image not readable: {input_data}"
            )

        return image

    if isinstance(
        input_data,
        (bytes, bytearray),
    ):
        array = np.frombuffer(
            input_data,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Uploaded image bytes could not be decoded."
            )

        return image

    if isinstance(input_data, np.ndarray):
        image = input_data

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Expected color image with shape (H, W, 3)."
            )

        return image

    raise TypeError(
        "Unsupported input type for image."
    )


def get_embedding(
    input_data,
    det_size=(640, 640),
    providers=None,
    ctx_id=0,
    pick="best",
):
    app = get_face_app(
        det_size=det_size,
        providers=providers,
        ctx_id=ctx_id,
    )

    image = read_image_any(input_data)

    faces = app.get(image)

    if not faces:
        return None

    if pick == "largest":
        face = max(
            faces,
            key=lambda current_face: (
                current_face.bbox[2]
                - current_face.bbox[0]
            )
            * (
                current_face.bbox[3]
                - current_face.bbox[1]
            ),
        )

    else:
        face = max(
            faces,
            key=lambda current_face: (
                float(current_face.det_score)
                * (
                    (
                        current_face.bbox[2]
                        - current_face.bbox[0]
                    )
                    * (
                        current_face.bbox[3]
                        - current_face.bbox[1]
                    )
                )
            ),
        )

    embedding = face.embedding

    if embedding is None:
        return None

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    embedding_norm = norm(embedding)

    if embedding_norm < 1e-8:
        return None

    return embedding / embedding_norm


def cosine_similarity(
    embedding1,
    embedding2,
):
    if embedding1 is None or embedding2 is None:
        return None

    embedding1 = np.asarray(
        embedding1,
        dtype=np.float32,
    )

    embedding2 = np.asarray(
        embedding2,
        dtype=np.float32,
    )

    denominator = (
        norm(embedding1)
        * norm(embedding2)
    )

    if denominator < 1e-8:
        return None

    similarity = np.dot(
        embedding1,
        embedding2,
    ) / denominator

    return float(similarity)


def verify_face_match(
    id_input,
    selfie_input,
    threshold=0.35,
    det_size=(640, 640),
    providers=None,
):
    id_embedding = get_embedding(
        id_input,
        det_size=det_size,
        providers=providers,
    )

    selfie_embedding = get_embedding(
        selfie_input,
        det_size=det_size,
        providers=providers,
    )

    if id_embedding is None:
        return {
            "ok": False,
            "similarity": None,
            "decision": "NO_FACE",
            "reason": "Face was not detected in the ID image.",
        }

    if selfie_embedding is None:
        return {
            "ok": False,
            "similarity": None,
            "decision": "NO_FACE",
            "reason": (
                "Face was not detected in the "
                "live-video frame."
            ),
        }

    similarity = cosine_similarity(
        id_embedding,
        selfie_embedding,
    )

    if similarity is None:
        return {
            "ok": False,
            "similarity": None,
            "decision": "INVALID_EMBEDDING",
            "reason": (
                "Face embeddings could not be compared."
            ),
        }

    decision = (
        "MATCH"
        if similarity >= threshold
        else "NO_MATCH"
    )

    return {
        "ok": True,
        "similarity": similarity,
        "threshold": threshold,
        "decision": decision,
    }


def threshold_from_scores(
    scores,
    labels,
    target_far=1e-3,
):
    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    labels = np.asarray(
        labels,
        dtype=np.int32,
    )

    impostor_scores = scores[
        labels == 0
    ]

    if len(impostor_scores) == 0:
        raise ValueError(
            "No impostor scores with label 0 were provided."
        )

    threshold = float(
        np.quantile(
            impostor_scores,
            1.0 - target_far,
        )
    )

    return threshold
```
