# 运行配置

cluster.json：主机路径及资源；models.json：模型身份；vllm_models.sh：服务参数；evaluations.json：通用评测预设；jobs.json：专用作业索引。

不在此存放凭证、生成数据或模型权重。专用作业的 #SBATCH 仍支持直接 sbatch，统一入口以 jobs.json 中的同值资源为默认，并支持命令行覆盖；修改默认资源时两处需保持一致。

使用方式和兼容边界见 [运行与 Slurm 说明](../docs/JOB_LAYOUT.md)。
