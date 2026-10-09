# Patch provenance

## UniLab

- Base: `d540f620b531a1b5100586ce652ae070e016ad60`
- Optimized commit: `139524ac893efecf76feaf827e7236cb77307b33`
- Expected optimized tree: `7c9e8bd42c384cf72e418c013af85152344464a6`
- Patch: `UniLab/0001-perf-distributed-enable-RoCE-dual-Spark-training.patch`

```bash
git clone https://github.com/Motphys/UniLab.git
git -C UniLab checkout d540f620b531a1b5100586ce652ae070e016ad60
git -C UniLab am ../patches/UniLab/*.patch
test "$(git -C UniLab rev-parse HEAD^{tree})" = \
  7c9e8bd42c384cf72e418c013af85152344464a6
```

## unilab-rl

- Base: `869155d0524838740de7723af81204818aa51a70`
- Optimized commit: `385a69f6d74bcfd9453bd2ed0c2d0687ae6c5556`
- Expected optimized tree: `253ad30993d8cdbd7209e4c368dda949f652f3c1`
- Patch: `unilab_rl/0001-perf-dp-optimize-dual-node-CUDA-graph-synchronizatio.patch`

```bash
git clone https://github.com/unilabsim/unilab_rl.git
git -C unilab_rl checkout 869155d0524838740de7723af81204818aa51a70
git -C unilab_rl am ../patches/unilab_rl/*.patch
test "$(git -C unilab_rl rev-parse HEAD^{tree})" = \
  253ad30993d8cdbd7209e4c368dda949f652f3c1
```

Patch files were produced with `git format-patch -1`, so commit messages,
authorship and the full tested diff are preserved.
