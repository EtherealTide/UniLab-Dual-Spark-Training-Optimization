# TorchEnv patch provenance

两个 patch 都是从用户指定的 dual-spark commit 生成的 cumulative binary diff，按
`git apply` 应用后得到本次集成 tree。它们包含 main 的 TorchEnv 迁移以及保留/修正
dual-spark 外部多节点拓扑所需的冲突决策。

## UniLab

起点：`d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3`
patch：`UniLab/0001-merge-torchenv-main-dual-spark.patch`
集成 commit：`d71875c0dc2bf8a054514b69547502664eb7b2e0`
最终 tree：`a3d3da9220544e7fb38b0c838c0e24cd6d775ed3`

## unilab-rl

起点：`a3ed997d5c25ff708d674778782bc1be08a53e15`
patch：`unilab_rl/0001-merge-torchenv-main-dual-spark.patch`
集成 commit：`a7f82c9476eaa61e97bf9ec6dd725f9a055ceb40`
最终 tree：`c55bfad35393d05be90077a6023581cdfa80abc3`

验证：

```bash
git -C UniLab checkout d2fef27e5a6786695cef58b57bc6fd8bbe84e7e3
git -C UniLab apply ../patches/UniLab/0001-merge-torchenv-main-dual-spark.patch
git -C UniLab write-tree  # a3d3da9220544e7fb38b0c838c0e24cd6d775ed3

git -C unilab_rl checkout a3ed997d5c25ff708d674778782bc1be08a53e15
git -C unilab_rl apply ../patches/unilab_rl/0001-merge-torchenv-main-dual-spark.patch
git -C unilab_rl write-tree  # c55bfad35393d05be90077a6023581cdfa80abc3
```
