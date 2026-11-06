#!/usr/bin/env bash
# Sourced by run_eval.sh; model settings preserve original condition semantics.
case "${MODEL_KIND}" in
    qwen25|qwen3)
        if [[ "${MODE}" == sufficient && "${QWEN_USE_FAST}" != "true" && "${QWEN_USE_FAST}" != "false" ]]; then
            echo "QWEN_USE_FAST must be true or false, got: ${QWEN_USE_FAST}" >&2
            exit 2
        fi
        MAX_MODEL_LEN=32768
        NUM_FRAMES=32
        TP=2
        MEDIA_MODE=frames
        MM_LIMIT='{"image": 32, "video": 0}'
        EXTRA_SERVER_ARGS=(--mm-processor-kwargs '{"max_pixels": 75264}')
        if [[ "${MODE}" == sufficient ]]; then
            EXTRA_SERVER_ARGS=(--mm-processor-kwargs "{\"max_pixels\":75264,\"use_fast\":${QWEN_USE_FAST}}")
        fi
        ;;
    llava)
        MAX_MODEL_LEN=32768
        NUM_FRAMES=32
        TP=1
        MEDIA_MODE=video
        MM_LIMIT='{"image": 0, "video": 1}'
        EXTRA_SERVER_ARGS=(--media-io-kwargs '{"video": {"num_frames": 32}}')
        if [[ "${MODE}" == merged ]]; then
            MEDIA_MODE=frames
            MM_LIMIT='{"image": 32, "video": 0}'
            EXTRA_SERVER_ARGS=()
        fi
        ;;
    internvl)
        MAX_MODEL_LEN=40960
        NUM_FRAMES=32
        TP=1
        MEDIA_MODE=frames
        MM_LIMIT='{"image": 32, "video": 0}'
        EXTRA_SERVER_ARGS=(
            --trust-remote-code
            --mm-processor-kwargs '{"max_dynamic_patch": 3}'
        )
        ;;
    *)
        echo "Unknown MODEL_KIND=${MODEL_KIND}" >&2
        exit 2
        ;;
esac
