# Release acceptance

No result is considered reproduced by a config-only or mocked run. Record command, UTC time, host/GPU inventory, peak memory, effective batch, throughput, exit status, and output hashes for every real smoke.

## Static gate

Run `bash scripts/acceptance/run_static.sh`. All shells, Python sources, tests, configs, path/secret scans, nested repository checks, and artifact-size checks must pass.

## Isolated download gate

Use an empty temporary ModelScope home, resolve every group, and retain the resolver log. Verification must cover all SHA-256 values, metadata tasks/records, FAISS vectors, and action-store references. Repeat once through `TRACEFLOW_ASSET_ROOT` with network disabled.

## H100 40 GB MIG gate

- LIBERO: each suite at one task × one episode, followed by the four-suite aggregator.
- LIBERO-Plus-10: one selected variant × one episode.
- Arena: each of the four entrypoints in single-GPU debug mode, batch 2 / env 2, one episode; verify a real Upper, Lower, and action call.
- TraceBankStack: LIBERO-10 and Transferring collector plus success/failure/joint round 1; rerun to exercise resume.

For every row verify videos, JSONL/TSV records, aggregate result, bank files, round provenance, and the second-run resume behavior.

## Publication gate

After ModelScope upload, rerun the isolated download smoke from a new cache. After GitHub push, clone the public repository into a new temporary directory, run setup preflight, static acceptance, config printing, and the real smoke matrix. The code repository must contain one root commit and no prior project history.

