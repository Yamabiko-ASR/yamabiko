# third_party/llama.cpp/

Vendored `llama-server.exe` + CUDA DLLs. See
[`../README.md`](../README.md) for the overall vendoring policy and the
binary-refresh recipe.

- **Upstream:** https://github.com/ggml-org/llama.cpp
- **Pinned commit + build flags:** see [`COMMIT.txt`](./COMMIT.txt)
- **License:** MIT, see [`LICENSE.txt`](./LICENSE.txt)

Build flags this project targets (CUDA, fp16, shared/sidecar-DLL
build -- must match `COMMIT.txt`):

```
-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DBUILD_SHARED_LIBS=ON
```

CUDA arch list: project defaults (`CMAKE_CUDA_ARCHITECTURES` unset)
under CUDA 12.9, which resolves to `50-virtual 61-virtual 70-virtual
75-virtual 80-virtual 86-real 89-real 120a-real 121a-real`. Adjust per
host and update `COMMIT.txt` to match if you deviate.

`BUILD_SHARED_LIBS=ON` is load-bearing: a static build produces no
`ggml*.dll` / `llama.dll` / `mtmd.dll` sidecars, breaking the layout
that `installer/llama-manifest.txt` and the watchdog spawn require.
