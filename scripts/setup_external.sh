#!/usr/bin/env bash
# clone the repositories of the pre-trained denoisers at the commits we used, and apply our patches.
#   mdlm      the diffusion transformer, its data pipeline and the released openwebtext model
#   sedd      the absorbing sedd models
#   maskgit   maskgit-pytorch: the imagenet transformer, its vqgan and the inception metrics
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
dest=${DOMINO_EXTERNAL:-$repo_dir/external}
mkdir -p "$dest"

fetch () {   # fetch <url> <name> <commit> [patch]
  [ -d "$dest/$2" ] && return
  git clone -q "$1" "$dest/$2"
  git -C "$dest/$2" checkout -q "$3"
  [ -z "${4:-}" ] || git -C "$dest/$2" apply "$repo_dir/patches/$4"
}

fetch https://github.com/kuleshov-group/mdlm mdlm c112c526d193436838c98d81455ee51f90309470 mdlm.patch
fetch https://github.com/louaaron/Score-Entropy-Discrete-Diffusion sedd 0605786da5ccb5747545e26d66fdf477187598b6 sedd.patch
fetch https://github.com/llvictorll/Maskgit-pytorch maskgit b0b2b3cc11cffd0b159f22dc1c6e73a7e8b53db3
echo "external repositories ready in $dest"
