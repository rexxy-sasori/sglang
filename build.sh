#!/bin/bash

# Build script for SGLang Docker images
# Usage: ./build.sh <dockerfile_path> <image_tag> [--no-cache]
# Example: ./build.sh docker/dev.Dockerfile src-session-isolation-v0.1.0
# Example: ./build.sh docker/dev.Dockerfile src-session-isolation-v0.1.0 --no-cache

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_PATH="${1:-}"
IMAGE_TAG="${2:-}"
CACHE_OPTION="${3:-}"
REGISTRY="harbor.xa.xshixun.com:7443/hanfeigeng/lmsysorg/sglang"
BASE_IMAGE="harbor.xa.xshixun.com:7443/hanfeigeng/lmsysorg/sglang:kv-cache-logging-dev-otel-0.8"

# Show usage if arguments are missing
if [ -z "$DOCKERFILE_PATH" ] || [ -z "$IMAGE_TAG" ]; then
    echo "Usage: $0 <dockerfile_path> <image_tag> [--no-cache]"
    echo ""
    echo "Examples:"
    echo "  $0 docker/dev.Dockerfile src-session-isolation-v0.1.0"
    echo "  $0 docker/Dockerfile src-session-isolation-v0.1.0 --no-cache"
    echo ""
    echo "Available Dockerfiles in docker/:"
    ls -1 "$SCRIPT_DIR/docker/Dockerfile"* 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "$SCRIPT_DIR/$DOCKERFILE_PATH" ]; then
    echo "Error: Dockerfile not found: $SCRIPT_DIR/$DOCKERFILE_PATH"
    exit 1
fi

# Parse cache option
NO_CACHE=""
if [ "$CACHE_OPTION" = "--no-cache" ]; then
    NO_CACHE="--no-cache"
    echo "Cache disabled: building without cache"
fi

echo "Building image..."
echo "  Dockerfile: $DOCKERFILE_PATH"
echo "  Tag: $REGISTRY:$IMAGE_TAG"
echo "  Base Image: $BASE_IMAGE"
echo "  Platform: linux/amd64"
echo ""

docker buildx build \
    --platform=linux/amd64 \
    $NO_CACHE \
    -f "$SCRIPT_DIR/$DOCKERFILE_PATH" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$REGISTRY:$IMAGE_TAG" \
    "$SCRIPT_DIR"

echo ""
echo "Build complete: $REGISTRY:$IMAGE_TAG"
echo ""
echo "To push the image, run:"
echo "  docker push $REGISTRY:$IMAGE_TAG"
