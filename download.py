from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/paligemma-3b-pt-224",
    local_dir=r"paligemma-3b-pt-224",
    local_dir_use_symlinks=False,
)
print("done")
