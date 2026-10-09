# Patch provenance

本目录以公开 `feat/dual-spark` HEAD 为唯一 checkout 起点，再通过有顺序的 cumulative
patch 重建正式实验代码。patch 使用 `git diff --binary` 生成，按文件名顺序应用。

## UniLab

- 公开分支：`feat/dual-spark`
- 分支 pin：`d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3`
- patch 1：实验必需的 off-policy runtime log interval 修复；对应后续提交
  `d540f620b531a1b5100586ce652ae070e016ad60`
- patch 2：RoCE launcher、文档和测试；对应优化提交
  `139524ac893efecf76feaf827e7236cb77307b33`
- Benchmark tree after patches 1-2: `7c9e8bd42c384cf72e418c013af85152344464a6`.
- Patch 3: post-benchmark native launcher TTY/lifecycle/logging fixes. An isolated
  rank0 output PTY preserves the existing SAC/FlashSAC Rich Live panel; PPO
  retains its original RSL-RL logger. No custom logger or algorithm changes.
- Current runtime tree after patches 1-3:
  `bfde7cf0e9be82156124c5d58e84df39ceb66674` (`UNILAB_RUNTIME_TREE`).
- Validation: 19 focused regression tests; existing MuJoCo PPO integration
  tests; a real two-Spark SAC run through a TTY with original Rich output; and a real SSH
  remote-path failure which prints the error and terminates rank0. No new
  500-iteration performance matrix was run for patch 3. Upstream full PR gates
  are still required before creating/updating a UniLab PR.

```bash
git clone https://github.com/Motphys/UniLab.git
git -C UniLab checkout d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3
git -C UniLab apply ../patches/UniLab/*.patch
git -C UniLab add -A
test "$(git -C UniLab write-tree)" = \
  bfde7cf0e9be82156124c5d58e84df39ceb66674
```

## unilab-rl

- 公开分支：`feat/dual-spark`
- 分支 pin：`a3ed997d5c25ff708d674778782bc1be08a53e15`
- patch 1：集成正式实验采用的 main runtime；测试 main pin 为
  `c45846467ec5a9470412532eb03b4f99da41d605`，集成结果 tree 为
  `278433647fb12d09da4e3dddabadc3440c86495e`
- patch 2：graph 内 DP、持久 gradient bucket、collective 融合和 finite gate；对应
  优化提交 `385a69f6d74bcfd9453bd2ed0c2d0687ae6c5556`
- 最终预期 tree：`253ad30993d8cdbd7209e4c368dda949f652f3c1`

```bash
git clone https://github.com/unilabsim/unilab_rl.git
git -C unilab_rl checkout a3ed997d5c25ff708d674778782bc1be08a53e15
git -C unilab_rl apply ../patches/unilab_rl/*.patch
git -C unilab_rl add -A
test "$(git -C unilab_rl write-tree)" = \
  253ad30993d8cdbd7209e4c368dda949f652f3c1
```

为什么不直接把 `869155d` 写成分支版本：它是本地实验整合提交，不是远程
`feat/dual-spark` HEAD。公开分支 pin 始终是上面的 `a3ed997d...`；`869155d` 仅作为
实验 provenance 记录在 `versions.env`。
