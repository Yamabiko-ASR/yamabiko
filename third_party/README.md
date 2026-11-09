# third_party/

Vendored third-party binaries that Yamabiko ships with, but does not build
from source as part of its own build pipeline.

## Vendoring policy

- **Pinned commits, not floating branches.** Every vendored component has a
  `COMMIT.txt` (or equivalent manifest) recording the exact upstream
  commit/tag it was built from, plus the host and build flags used. Any
  contributor should be able to rebuild a bit-identical binary from the
  recorded inputs.
- **Binaries live outside git.** The actual `.exe` / `.dll` / `.so` files
  are multi-hundred-megabyte artifacts and are `.gitignore`d. Only the
  manifest, license, and README for each component are tracked.
- **Licenses travel with the binary.** Each vendored component has its
  upstream `LICENSE.txt` copied into its subdirectory so downstream users
  of the installed Yamabiko bundle receive the required license notices.
- **The maintainer drops binaries in before building the installer.**
  Vendored binaries are not fetched automatically during a normal
  developer workflow; `pip install -e .` and the test suite don't need
  them. They are required for the packaged installer build.

## What lives here

### `llama.cpp/`

Vendored `llama-server.exe` plus the CUDA runtime DLLs it depends on.
Yamabiko's translator backend speaks OpenAI-compatible HTTP to a local
llama-server instance; shipping a known-good build decouples Yamabiko
releases from whatever llama.cpp version happens to be on a user's
machine.

Tracked files:

- `llama.cpp/README.md`: pointer to upstream + component-specific notes
- `llama.cpp/COMMIT.txt`: pinned upstream commit, build host, and flags
- `llama.cpp/LICENSE.txt`: upstream MIT license (copy of
  https://github.com/ggml-org/llama.cpp/blob/master/LICENSE)

Gitignored files (dropped in by the maintainer at build time):

- `llama.cpp/llama-server.exe`
- `llama.cpp/*.dll`: CUDA/ggml runtime DLLs the executable loads
- The MSVC C/C++ runtime + OpenMP DLLs (`msvcp140.dll`, `vcruntime140.dll`,
  `vcruntime140_1.dll`, `vcomp140.dll`) are **auto-sourced** from the build
  host's installed Visual Studio redist by `installer/vendor-vcredist.ps1`
  (run by `build.ps1` / `rebuild-fast.ps1` before the manifest assert); you
  do **not** copy these by hand. The full required set is enumerated in
  `installer/llama-manifest.txt`.

## Refreshing a vendored binary

Maintainer-side procedure:

1. Check out the upstream repository at the commit/tag you want to pin.
2. Build on the reference host with the flags recorded in the component's
   `COMMIT.txt` (adjust if you deliberately deviate, but then update the
   manifest to match).
3. Drop the resulting `.exe` + CUDA/ggml `.dll`s into the component
   directory, overwriting whatever is there. (The MSVC runtime + OpenMP
   DLLs are auto-sourced by `installer/vendor-vcredist.ps1` at build time;
   don't copy those manually. If the CUDA major version of the build
   changes, update the DLL names in `installer/llama-manifest.txt` to
   match, e.g. `cublas64_13.dll` → `cublas64_12.dll`.)
4. Update the component's `COMMIT.txt`:
   - `commit:` 40-char SHA of the upstream commit
   - `tag:` release tag if applicable, else `untagged`
   - `built:` `YYYY-MM-DD` of the build
   - `host:` OS / GPU / CUDA / cmake versions
   - `flags:` and `arch:` if you deviated from the recorded defaults
5. Run the Yamabiko installer build and verify the binary launches and
   serves the expected OpenAI-compatible endpoint before cutting a
   release.

## Responsibility

The maintainer owns producing and dropping the
vendored binaries. Contributors do not need to do this to run the test
suite or develop against Yamabiko day-to-day.
