#!/usr/bin/env python3
# DeepSeek-V4-Flash(304B MoE)LoRA SFT · Megatron-Bridge · 8×B300(sm103)
#
# 数据用 prepare_dsv4_data.py + pack_dsv4_windows.py 的产出,提供器是同目录的
# dsv4_sft_data.py(独立副本,不依赖别的项目目录)。
#
# ⛔ 五条硬约束(不是偏好,是模型结构、内核或数学决定的):
#   ① TP 必须 = 1 —— 混合注意力不支持张量并行(官方配方和 ms-swift 口径一致)
#   ② PP > 1 时必须给显式的流水线切分表 —— 逐层压缩比不同,不能均匀切
#   ③ LoRA 挂点不能用库默认值 —— 默认是 linear_qkv,而 MLA 根本没有这个模块
#   ④ CP > 1 时必须 PACKED=1 + cp_partition_mode='contiguous' + DSv4 专用 forward_step
#   ⑤ 正式训练的 loss 必须按全局监督 token 总数归一
#      (calculate_per_token_loss=True)。cp_local 只允许显式诊断 ablation，
#      因为 contiguous 切分下监督 token 全挤在窗口后半，按各 CP 段自己归一目标会错。
#
# 📌 骨架仍是 2026-08-04 跑完 428 步的那套配方(CP=4 / 65536 / 打包),
#    外加 2026-08-06 的四处修正(那 428 步都不带,评测仅作诊断,正式结论以重跑为准):
#      · loss 归一化(⑤)
#      · WEIGHT_DECAY 真正生效 —— 此前被调度器覆写成 helper 默认的 0.033
#      · SEED 显式化 —— 此前 SEED=0 静默失效(模型 5678 / 数据 42)
#      · save_rng 默认开、beta2 0.999→0.98(想复现旧跑:ADAM_BETA2=0.999 SAVE_RNG=0)
#    调试开关(DEBUG_* / OFFLOAD)默认全关,开了才改变行为 —— 见文件末尾的说明。
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_sft_data import DSv4PreTokenizedProvider  # noqa: E402

from megatron.bridge import AutoBridge  # noqa: E402
from megatron.bridge.peft.lora import LoRA  # noqa: E402
from megatron.bridge.recipes.common import _peft_common  # noqa: E402
from megatron.bridge.recipes.utils.optimizer_utils import (  # noqa: E402
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig  # noqa: E402
from megatron.bridge.training.config import MockGPTDatasetConfig  # noqa: E402
from megatron.bridge.training.finetune import finetune  # noqa: E402
from megatron.bridge.training.gpt_step import forward_step as _gpt_forward_step  # noqa: E402
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig  # noqa: E402
from megatron.bridge.training.pretrain import pretrain  # noqa: E402

# ⛔ CP>1 必须用 DSv4 专用 step:只有它把 cp_partition_mode 注入 batch,再由
#    get_packed_seq_params 转发进 PackedSeqParams。用通用 step 会在前向里报
#    "DSv4 THD CP requires a contiguous CP partition."(2026-08-04 实测)
if int(os.environ.get("CP", "4")) > 1:
    from megatron.bridge.models.deepseek.deepseek_v4_step import (  # noqa: E402
        forward_step as _dsv4_forward_step,
    )
    forward_step = _dsv4_forward_step
else:
    forward_step = _gpt_forward_step

MODEL_DIR = os.environ.get(
    "DSV4_MODEL",
    "/home/work/migoo_ai_public/posttrain/post-train/models/DeepSeek-V4-Flash-0731")


def _env(k, d):  return os.environ.get(k, d)
def _envi(k, d): return int(os.environ.get(k, d))
def _envf(k, d): return float(os.environ.get(k, d))


# DSv4 的 MLA 里没有 linear_qkv(那是普通 GQA 的名字)。实际的线性层是这几个:
#   linear_q_down_proj / linear_q_up_proj   Q 的低秩两段
#   linear_kv_proj                          KV 压缩
#   linear_proj                             输出投影
# ⛔ 用库默认的 ["linear_qkv", ...] 会**一个 Q/KV 都挂不上**,而且不报错,
#    只是 "Adding lora to" 的计数变少 —— 起训后必须数这个数(期望 4 × 43 = 172)。
ATTN_TARGETS = ["linear_q_down_proj", "linear_q_up_proj", "linear_kv_proj", "linear_proj"]
MLP_TARGETS = ["linear_fc1", "linear_fc2"]


def build_config():
    # ── 调试开关(默认全关,生产路径完全不受影响)
    random_init = _env("DEBUG_RANDOM_INIT", "0") == "1"       # 不加载底模档,随机初始化
    random_lora = random_init and (_env("DEBUG_RANDOM_LORA", "1") == "1")
    debug_mock_data = _env("DEBUG_MOCK_DATA", "1" if random_init else "0") == "1"
    debug_layers = _envi("DEBUG_NUM_LAYERS", 0)               # 只建前 N 层

    data_dir = os.environ.get("DSV4_DATA", "")
    output_dir = os.environ["OUTPUT_DIR"]
    megatron_ckpt = os.environ.get("MEGATRON_CKPT", "")

    ep = _envi("EP", 8)
    tp = _envi("TP", 1)
    pp = _envi("PP", 1)
    cp = _envi("CP", 4)
    # ⛔ 65536 不是随手挑的:98304 能一条不丢,但 CP=4 下装不进(见文档 §6.4)。
    #    65536 丢 19/1804 条(1.05%)。想要 98304 得先上 CP=8,没验过。
    seq_len = _envi("SEQ_LEN", 65536)
    mbs = _envi("MBS", 1)
    gbs = _envi("GBS", 8)                  # 打包后 1715 窗口 ÷ 8 ≈ 214 步/epoch
    train_iters = _envi("TRAIN_ITERS", 10)
    warmup_iters = _envi("WARMUP_ITERS", max(1, math.ceil(0.03 * train_iters)))
    lr = _envf("LR", 1e-4)                 # ms-swift 的 DSv4 LoRA 参考值
    loss_reduction = _env("LOSS_REDUCTION", "global").strip().lower()
    if loss_reduction not in ("global", "cp_local"):
        raise ValueError(
            f"unsupported LOSS_REDUCTION={loss_reduction!r}; expected global or cp_local"
        )
    mask_mode = _env("SFT_MASK_MODE", "source").strip().lower()
    if mask_mode not in ("source", "all_nonpad"):
        raise ValueError(
            f"unsupported SFT_MASK_MODE={mask_mode!r}; expected source or all_nonpad"
        )
    max_samples = _envi("SFT_MAX_SAMPLES", 0)
    max_windows = _envi("SFT_MAX_WINDOWS", 0)

    assert tp == 1, "⛔ DSv4 的混合注意力不支持 TP,TP 必须是 1"

    # ⛔ LoRA 训练**必须**给一个 Megatron 格式的底模档。Megatron-Bridge 的
    #    config.py:1358 有硬校验:`PEFT requires a pretrained checkpoint path`,
    #    不给会在起训时直接报这个错(不是静默随机初始化,是当场停)。
    #    档从哪来:先跑一次 convert_dsv4_ckpt.py 把 HF 权重转过去。
    if not random_init:
        assert megatron_ckpt, (
            "⛔ 没给 MEGATRON_CKPT。LoRA 训练必须要一个 Megatron 格式的底模档,\n"
            "   先转一次(8 卡,一次性):\n"
            "     source env_dsv4.sh\n"
            "     $DSV4_PY -m torch.distributed.run --nproc_per_node 8 convert_dsv4_ckpt.py")
        assert data_dir, "⛔ 没给 DSV4_DATA(训练数据目录)"

    bridge = AutoBridge.from_hf_pretrained(MODEL_DIR, trust_remote_code=True)
    # 权重由上面那个 Megatron 档提供,所以这里不用桥接再灌一遍 HF 权重
    m = bridge.to_megatron_provider(load_weights=False)

    m.expert_model_parallel_size = ep
    m.expert_tensor_parallel_size = 1
    m.tensor_model_parallel_size = tp
    m.pipeline_model_parallel_size = pp
    m.context_parallel_size = cp
    m.sequence_parallel = False          # SP 依赖 TP>1,这里恒为 False
    m.pipeline_dtype = torch.bfloat16
    m.seq_length = seq_len

    # MTP:LoRA SFT 默认关(省一个巨大的第二 logits 头)。想开设 MTP=1。
    if _env("MTP", "0") == "0":
        m.mtp_num_layers = None
        # 桥接按 num_nextn_predict_layers 给逐层压缩比表多补了一项(43+1=44)。
        # 关掉 MTP 后表长必须等于 num_layers(43),否则每一步的 FLOPs 统计里
        # `len(compress_ratios) != num_layers` 会抛 ValueError —— 而那个统计在
        # 训练循环里每步都调、没有开关,发作时机是权重加载完跑完第一步。
        # 官方的 deepseek_v4_flash_no_mtp_sft_config() 同样要裁这一刀。
        ratios = getattr(m, "csa_compress_ratios", None)
        if ratios is not None and len(ratios) > m.num_layers:
            m.csa_compress_ratios = list(ratios)[: m.num_layers]
    else:
        m.mtp_loss_scaling_factor = _envf("MTP_LOSS_SCALE", 0.1)

    # PP>1 必须给切分表(逐层压缩比不同,不能均匀切)。一定要在上面的
    # MTP 开关之后生成，否则 MTP=0 时 layout 仍残留一个 mtp 层，最终配置
    # 校验会报 "Number of mtp layers in layout must match mtp_num_layers"。
    if pp > 1:
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
            set_deepseek_v4_pipeline_model_parallel_layout,
        )
        set_deepseek_v4_pipeline_model_parallel_layout(m)

    # ── DSv4 专有开关,取自官方 SFT 配方 deepseek_v4_flash_sft_config()
    from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
        deepseek_v4_supports_blackwell_fused_kernels,
    )
    m.transformer_impl = "transformer_engine"
    m.attention_backend = None                     # 让 TE 自己挑
    # ⛔ 长窗口的硬前提,别照抄官方配方的 False。
    #    关掉之后走 csa.py 的参考实现,里面 `einsum('sbhd,tbd->sbht')` 会物化
    #    一个「序列 × 序列 × 头数」的 fp32 稠密打分矩阵 —— O(序列²)。
    #    4096 窗口约 4GB(所以官方配方能用),98304 窗口要 **576 GiB**,一次申请就 OOM。
    #    官方配方设 False 是因为它配套的窗口只有 4096,不是因为"融合只是提速"。
    #    开这个需要 flash_mla + nvidia-cudnn-frontend[cutedsl](setup_dsv4_env.sh 会装)。
    #    ⚠️ 空字符串要当作"没设"处理 —— 启动器里 `export DSA_FUSION="${DSA_FUSION:-}"`
    #    会导出一个空值,直接 `== "1"` 判就成了 False,等于在最不该关的时候关掉了。
    _dsa = os.environ.get("DSA_FUSION", "").strip()
    m.apply_dsa_kernel_fusion = (_dsa == "1") if _dsa else (seq_len > 8192)

    # ── CP:DSv4 的 CSA 要求 THD 行按连续段切(zigzag 会被内核拒绝),
    #    而 thd 又要求数据是打包的 —— 三者必须同时满足,少一个就报错。
    if cp > 1:
        m.cp_partition_mode = "contiguous"
        assert debug_mock_data or _env("PACKED", "0") == "1", (
            "⛔ CP>1 必须配 PACKED=1(thd 格式)。先打包:\n"
            "     python3 pack_dsv4_windows.py --data-dir $DSV4_DATA --window $SEQ_LEN")

    # ⛔ 约束⑤:loss 按「全局监督 token 总数」归一,不能按各 CP 段自己的数。
    #    这批数据平均一个窗口约 1 条样本、监督 token(~881/窗口)全在样本尾部,
    #    contiguous 切 4 段后实测(1715 窗口):段0/1 恒为 0,段3 有 77% 全零。
    #    cp_local 时每段先各自求平均再等权相加 —— 全零段稀释梯度、少 token 段
    #    每个 token 被放大几十倍;而日志 loss 走的是另一条(正确的)聚合路,
    #    曲线上完全看不出来。Bridge 对自家 SFT 数据集有硬校验(config.py:1418
    #    "When finetuning with CP>1, calculate_per_token_loss must be True"),
    #    但自定义 provider 不在它的 isinstance 名单里,正式路径必须显式设。
    #    CP=1 也保持 global:各种 CP 的跑法目标一致,才能做 loss/梯度对拍。
    if loss_reduction == "cp_local" and cp > 1 and _env("ALLOW_UNSAFE_LOSS_ABLATION", "0") != "1":
        raise ValueError(
            "⛔ LOSS_REDUCTION=cp_local 在 CP>1 下仅允许显式 ablation: "
            "设置 ALLOW_UNSAFE_LOSS_ABLATION=1"
        )
    m.calculate_per_token_loss = loss_reduction == "global"

    m.apply_rope_fusion = True
    # B300 是 sm103(major=10)→ 这里会是 True,融合 mHC 打开
    m.use_fused_mhc = deepseek_v4_supports_blackwell_fused_kernels()
    # 索引器辅助 loss:官方 SFT 配方也是 0.0 + False(refs/A_官方配方…py:90-91)
    m.dsa_indexer_loss_coeff = 0.0
    m.dsa_indexer_use_sparse_loss = False

    # ── MoE
    m.moe_token_dispatcher_type = "alltoall"
    m.moe_aux_loss_coeff = _envf("MOE_AUX_LOSS", 0.0)
    m.moe_router_force_load_balancing = False

    # 交叉熵必须用 TE 融合版:原生实现会把 fp32 logits 全量物化
    # (seq_len × 129280 × 4B × 2),必 OOM。Hy3 和 qwen36 都栽过。
    m.cross_entropy_loss_fusion = True
    m.cross_entropy_fusion_impl = _env("CE_FUSION_IMPL", "te")

    # 重算:官方配方用 selective(只重算 moe_act 和 mhc),但那是给 **4096** 窗口的。
    # ⚠️ 开 full 有个副作用:第一遍前向在 torch.no_grad() 里跑,DSv4 的稀疏注意力
    #    会据此误判成"推理态",走进一个要物化 O(窗口²) 打分矩阵的分支 —— 这就是
    #    CP 必须开的真正原因,详见文档 §6.4。
    rc = _env("RECOMPUTE", "full")
    if rc == "selective":
        m.recompute_granularity = "selective"
        m.recompute_modules = ["moe_act", "mhc"]
        m.recompute_method = None
        m.recompute_num_layers = None
    elif rc == "full":
        m.recompute_granularity = "full"
        m.recompute_method = "uniform"
        m.recompute_num_layers = _envi("RECOMPUTE_LAYERS", 1)
    else:
        m.recompute_granularity = None
        m.recompute_method = None
        m.recompute_num_layers = None

    # 细粒度激活卸载(显存实在不够时的最后一招,会拖慢步时)
    if _env("OFFLOAD", "0") == "1":
        m.fine_grained_activation_offloading = True
        m.offload_modules = [x for x in _env("OFFLOAD_MODULES", "attn_norm,mlp_norm").split(",") if x]

    m.bf16 = True
    m.params_dtype = torch.bfloat16
    m.cuda_graph_impl = "none"          # 官方配方显式关掉

    # 【调试】只建前 N 层 —— 排查显存/内核问题时用,能把加载时间从 10 分钟压到 1 分钟
    if debug_layers > 0:
        assert pp == 1, "DEBUG_NUM_LAYERS 只建议在 PP=1 的 smoke/debug run 使用"
        m.num_layers = debug_layers
        for name in ("csa_compress_ratios", "moe_layer_freq", "linear_attention_freq"):
            values = getattr(m, name, None)
            if isinstance(values, list):
                setattr(m, name, list(values)[:debug_layers])

    cfg = _peft_common()
    cfg.model = m

    # ── 官方配方的「robustness defaults」。_peft_common() 不带这些,漏了会出问题:
    #    enable_megatron_core_experimental 默认 False,而 DSv4 的混合注意力
    #    (dsv4_hybrid)整套走的都是 mcore 的实验通道,不开就建不出模型。
    cfg.dist.enable_megatron_core_experimental = True
    # 98k + PP 首步会触发 MoE/CUTLASS 的首次编译和大规模重算，可能超过
    # Megatron 默认的 10 分钟 process-group watchdog。初始超时单独放宽；
    # 首个 iteration 成功后可恢复到较短超时，避免真正的通信死锁久等。
    initial_timeout_minutes = _envi("DIST_TIMEOUT_MINUTES", 10)
    timeout_seconds_after_init = _envi("DIST_TIMEOUT_SECONDS_AFTER_INIT", 0)
    if initial_timeout_minutes <= 0:
        raise ValueError("DIST_TIMEOUT_MINUTES 必须为正整数")
    if timeout_seconds_after_init < 0:
        raise ValueError("DIST_TIMEOUT_SECONDS_AFTER_INIT 不能为负数")
    cfg.dist.distributed_timeout_minutes = initial_timeout_minutes
    cfg.dist.distributed_timeout_seconds_after_init = timeout_seconds_after_init or None
    cfg.ddp.use_megatron_fsdp = False
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # ── LoRA
    targets = list(ATTN_TARGETS)
    if _env("LORA_TARGET_MLP", "0") == "1":
        targets += MLP_TARGETS
    stride = _envi("LORA_LAYER_STRIDE", 0)
    if stride > 1:
        off = _envi("LORA_LAYER_OFFSET", 0)
        picked = list(range(off, m.num_layers, stride))
        targets = [f"*.layers.{i}.*.{s}" for i in picked for s in targets]
    else:
        picked = None

    # r=64/alpha=128 是 2026-08-04 实跑用的值(ms-swift 参考配置是 16/32,没实跑过)
    lora_cfg = LoRA(
        target_modules=targets,
        dim=_envi("LORA_R", 64),
        alpha=_envi("LORA_ALPHA", 128),
        dropout=_envf("LORA_DROPOUT", 0.05),
        dropout_position="pre",
        lora_A_init_method="xavier",
        lora_B_init_method="zero",
    )

    if random_lora:
        # 【调试】在随机初始化的模型上挂 LoRA。Bridge 默认的 peft hook 会先强制
        # 加载 pretrained checkpoint(那要 10 分钟读 568GB),这里绕开它 ——
        # 排查内核/显存问题时能把一轮迭代从 40 分钟压到几分钟。
        cfg.peft = None

        def _apply_lora_without_checkpoint(model):
            if int(os.environ.get("RANK", "0")) == 0:
                print("[dsv4-debug] 在随机初始化的模型上挂 LoRA,跳过 checkpoint 加载", flush=True)
            transformed = lora_cfg(model, training=True)
            lora_cfg.set_params_to_save(transformed)
            return transformed

        m.register_pre_wrap_hook(_apply_lora_without_checkpoint)
        cfg._debug_peft = lora_cfg
    elif random_init:
        cfg.peft = None
        cfg._debug_peft = None
    else:
        cfg.peft = lora_cfg
        cfg._debug_peft = lora_cfg

    # ⛔ 权重衰减必须同时喂给 optimizer 和 scheduler:mcore 的调度器每一步都把
    #    参数组里的 weight_decay 覆写成自己的 start/end 插值
    #    (optimizer_param_scheduler.py:310),只改 cfg.optimizer.weight_decay
    #    活不过第一步 —— 8-04 那 428 步就这样在 helper 写死的 0.033 下跑完了,
    #    环境变量给的 0 从没生效过。
    wd = _envf("WEIGHT_DECAY", 0.0)
    # beta2 默认 0.98:Bridge PEFT 配方的通用值("Common for fine-tuning")。
    # 8-04 用的 0.999 对 ~428 步的短训偏保守(二阶矩要 ~1000 步才跟得上),
    # 想复现旧跑设 ADAM_BETA2=0.999。
    opt_cfg, sched_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=warmup_iters, lr_decay_iters=train_iters,
        max_lr=lr, min_lr=_envf("MIN_LR", 0.0),
        adam_beta2=_envf("ADAM_BETA2", 0.98),
        weight_decay=wd, start_weight_decay=wd, end_weight_decay=wd,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = sched_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    for k in ("main_grads_dtype", "main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        setattr(cfg.optimizer, k, torch.float32)

    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
        autocast_enabled=False, grad_reduce_in_fp32=True,
    )
    # global token loss 要先求和再按全局监督 token 数归一;cp_local 保留旧的
    # collective 平均行为,仅用于诊断 A/B,不能当作正式推荐配置。
    cfg.ddp.average_in_collective = loss_reduction != "global"
    cfg.ddp.check_for_nan_in_grad = True      # ⚠️ DSv4 出过 rope 越界导致 loss NaN,别关

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = gbs
    cfg.train.micro_batch_size = mbs

    # SEED 语义:环境变量**设了就用**(0 也是有效种子);没设走库默认并在横幅回显。
    # 旧写法 `if seed:` 让 SEED=0 静默失效 —— 模型 RNG 落回 Bridge 默认 5678、
    # 数据洗牌落回 42,看着控制了种子,其实什么都没改。
    seed_env = os.environ.get("SEED", "").strip()
    seed = int(seed_env) if seed_env else None
    if seed is not None:
        cfg.rng.seed = seed

    if debug_mock_data:
        # 【调试】用 mcore 自带的随机数据集,把数据这一环从排查里摘出去
        cfg.dataset = MockGPTDatasetConfig(
            seq_length=seq_len,
            random_seed=seed if seed is not None else 1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            num_dataset_builder_threads=1,
            split="9999,8,2",
            data_sharding=True,
            dataloader_type="single",
            skip_getting_attention_mask_from_dataset=True,
            num_workers=_envi("NUM_WORKERS", 0),
        )
        cfg.tokenizer.tokenizer_type = "NullTokenizer"
        cfg.tokenizer.tokenizer_model = None
        cfg.tokenizer.vocab_size = m.vocab_size
        cfg.tokenizer.make_vocab_size_divisible_by = m.make_vocab_size_divisible_by
        cfg.tokenizer.tensor_model_parallel_size = m.tensor_model_parallel_size
        cfg.tokenizer.rank = 0
        cfg.tokenizer.use_tokenizer_vocab_size = False
    else:
        cfg.dataset = DSv4PreTokenizedProvider(
            seq_length=seq_len, data_dir=data_dir,
            packed=_env("PACKED", "1") == "1",
            mask_mode=mask_mode,
            max_samples=max_samples,
            max_windows=max_windows,
            dataloader_type="single", num_workers=_envi("NUM_WORKERS", 2),
            **({"seed": seed} if seed is not None else {}),
        )
        cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
        cfg.tokenizer.tokenizer_model = MODEL_DIR

    cfg.checkpoint.pretrained_checkpoint = None if random_init else megatron_ckpt
    cfg.checkpoint.save = output_dir
    cfg.checkpoint.load = None if random_init else output_dir
    cfg.checkpoint.save_interval = train_iters + 1 if random_init else _envi("SAVE_INTERVAL", 43)
    cfg.checkpoint.most_recent_k = _envi("KEEP_CKPT", -1)
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.save_optim = False if random_init else (_env("SAVE_OPTIM", "1") == "1")
    # RNG 状态默认存:load=output_dir 是自动续训路径,不存的话断点续训后
    # dropout 的抽样序列和不中断的跑法对不上,数值无法逐位复现
    # (优化器状态、数据位置各有自己的恢复机制,不受这个开关影响)。
    cfg.checkpoint.save_rng = _env("SAVE_RNG", "1") == "1"

    cfg.validation.eval_iters = _envi("EVAL_ITERS", 0)
    cfg.validation.eval_interval = _envi("EVAL_INTERVAL", 1000)
    cfg.logger.log_interval = _envi("LOG_INTERVAL", 1)
    # ⛔ tensorboard 必须显式钉到输出目录 —— Bridge 的默认值跟着**当前工作目录**走
    #   (nemo_experiments/default/tb_logs)。2026-08-06 从仓库目录起训,tb 就写进了
    #   共享盘,共享盘一满 event 写盘线程 ENOSPC,把整个训练进程带崩(两次,
    #   第 77 步和第 84 步)。输出目录在哪,tb 就在哪。
    cfg.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")
    cfg.logger.wandb_project = _env("WANDB_PROJECT", "dsv4-flash-lora")
    cfg.logger.wandb_exp_name = _env("WANDB_EXP", "dsv4_lora")
    cfg.logger.wandb_entity = _env("WANDB_ENTITY", "") or None
    cfg.logger.wandb_save_dir = output_dir

    cfg._picked = picked
    cfg._debug_random_init = random_init
    cfg._debug_mock_data = debug_mock_data
    cfg._ablation_loss_reduction = loss_reduction
    cfg._ablation_mask_mode = mask_mode
    return cfg


def main():
    # LoRA 逐模块的 "Adding lora to: <名字>" 是 logger.info 级别,默认级别下**全被过滤**,
    # 于是 `grep -c` 数出来是 0 —— 看着像"一个都没挂上",其实挂得好好的。
    # 这里把 PEFT 那个 logger 单独调到 INFO,让挂点条数真的能数(期望 4 × 43 = 172)。
    import logging
    logging.getLogger("megatron.bridge.peft").setLevel(logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING)

    cfg = build_config()
    if int(os.environ.get("RANK", "0")) == 0:
        m = cfg.model
        print("[dsv4] DeepSeek-V4-Flash LoRA · Megatron-Bridge", flush=True)
        print(f"  并行: TP={m.tensor_model_parallel_size} PP={m.pipeline_model_parallel_size} "
              f"EP={m.expert_model_parallel_size} CP={m.context_parallel_size}", flush=True)
        print(f"  层数={m.num_layers}  窗口={m.seq_length}  MTP={m.mtp_num_layers}  "
              f"融合mHC={m.use_fused_mhc}  融合稀疏注意力={m.apply_dsa_kernel_fusion}  "
              f"重算={m.recompute_granularity}", flush=True)
        peft_cfg = getattr(cfg, "_debug_peft", cfg.peft)
        if peft_cfg is not None:
            print(f"  LoRA: r={peft_cfg.dim} alpha={peft_cfg.alpha} 挂点={peft_cfg.target_modules[:6]}"
                  f"{' …' if len(peft_cfg.target_modules) > 6 else ''}", flush=True)
        else:
            print("  LoRA: 【调试】关闭", flush=True)
        print(f"  批: mbs={cfg.train.micro_batch_size} gbs={cfg.train.global_batch_size} "
              f"iters={cfg.train.train_iters} lr={cfg.optimizer.lr}  "
              f"loss_reduction={cfg._ablation_loss_reduction} "
              f"mask={cfg._ablation_mask_mode} "
              f"全局token归一={m.calculate_per_token_loss}", flush=True)
        # wd 报 scheduler 的值 —— 它才是每步真正生效的(optimizer 字段会被它覆写)
        print(f"  优化: wd={cfg.scheduler.start_weight_decay} "
              f"beta2={cfg.optimizer.adam_beta2}  "
              f"种子: 模型RNG={cfg.rng.seed} 数据洗牌={getattr(cfg.dataset, 'seed', '—')}  "
              f"save_rng={cfg.checkpoint.save_rng}", flush=True)
        if getattr(cfg, "_debug_random_init", False):
            src = "【调试】随机初始化,不加载 checkpoint"
        else:
            src = (f"预转换的 Megatron 档 {cfg.checkpoint.pretrained_checkpoint}"
                   if cfg.checkpoint.pretrained_checkpoint else f"HF 权重直接流式加载 {MODEL_DIR}")
        print(f"  底模权重来源: {src}", flush=True)
        print(f"  数据来源: {'【调试】mock 随机数据集' if getattr(cfg, '_debug_mock_data', False) else '真实 SFT 数据'}",
              flush=True)
        # ⚠️ 这行**故意不写全** LoRA 那句日志的原文 —— 写全了自己会被
        #    `grep -c` 数进去,挂点计数永远至少是 1,看不出真挂了几个。
        print("  ⚠️ 起训后 5 分钟内必查:①LoRA 挂点条数(见文档 §7)②首步 loss 不是 NaN "
              "③显存 ④底模权重来源", flush=True)

    if getattr(cfg, "_debug_random_init", False):
        pretrain(cfg, forward_step_func=forward_step)
    else:
        finetune(cfg, forward_step_func=forward_step)


# ── 调试开关速查(默认全关,只在排查问题时用)────────────────────────────
#
#   DEBUG_RANDOM_INIT=1   不加载 568GB 底模档,随机初始化 —— 一轮迭代从 40 分钟压到几分钟
#   DEBUG_RANDOM_LORA=1   配合上面用:仍然挂 LoRA(绕开 Bridge 强制加载 ckpt 的 hook)
#   DEBUG_MOCK_DATA=1     用 mcore 自带的随机数据集,把数据这一环从排查里摘出去
#   DEBUG_NUM_LAYERS=8    只建前 8 层,快速试显存边界
#   OFFLOAD=1             细粒度激活卸载(显存不够的最后一招,会拖慢步时)
#
# 典型用法(2026-08-04 定位 §6.4 那个显存问题时就是这么跑的):
#   DEBUG_RANDOM_INIT=1 DEBUG_MOCK_DATA=1 DEBUG_NUM_LAYERS=8 \
#     SEQ_LEN=49152 CP=1 TRAIN_ITERS=1 bash run_dsv4.sh
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
