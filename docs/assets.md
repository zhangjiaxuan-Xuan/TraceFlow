# Asset release

The public ModelScope repository is organized as:

```text
common/libero/
libero/b6500_positive/
libero/b50_positive/
libero/ncpi_negative/
libero/libero10/{positive,negative}/
arena/extra8/
arena/{counting,occlusion}/
cl/layout.json
manifest.json
```

Only retrieval heads, metadata, FAISS indices, packed actions, gate/config metadata, and provenance belong there. Pi0.5 and PrediMem weights must not be staged.

`tools/build_release_assets.py` constructs this tree. For Full26 Arena data it reconstructs the selected suite vectors into new FAISS indices and repacks only referenced action chunks. This avoids publishing repeated multi-gigabyte Full26 metadata/index files. It then writes SHA-256, byte size, sample/vector/reference counts, task IDs, and the producer checkpoint identifier to `manifest.json`.

After inspecting the staging directory:

```bash
PYTHONPATH=src python tools/publish_modelscope.py /path/to/staged-assets
```

If neither `MODELSCOPE_API_TOKEN` nor `MODELSCOPE_TOKEN` is set, the publisher
prompts for the access token with terminal echo disabled. The token is used only
for the current process and is never persisted by TraceFlow. Folder upload uses
the ModelScope SDK's resumable cache, so rerunning the same command safely skips
files that were already committed.

For an isolation test, point `MODELSCOPE_CACHE` at an empty temporary directory, leave `TRACEFLOW_ASSET_ROOT` unset, resolve every manifest group, and run the smoke matrix. This proves that launchers use the snapshot returned by the API rather than a developer cache path.
