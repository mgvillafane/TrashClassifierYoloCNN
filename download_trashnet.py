from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="garythung/trashnet",
    repo_type="dataset",
    local_dir="/Volumes/workspace/default/trashnet_data"
)