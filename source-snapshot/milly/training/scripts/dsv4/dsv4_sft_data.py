#!/usr/bin/env python3
# DSv4 预切词 SFT 数据提供器 —— 喂给 Megatron-Bridge 的 gpt_step.get_batch。
#
# 📌 来源:2026-08-06 从 Hy3 项目快照过来的
#    (.../hy3_295b_lora/megatron_bridge/hy3_sft_data.py),做了两处改动:
#      ① pad 默认值 120002(Hy3 的)→ 1(DSv4 的 pad/eos),不用再靠 SFT_PAD_ID 兜
#      ② 类名 Hy3PreTokenizedProvider → DSv4PreTokenizedProvider
#    **这是一份独立副本,不是软链** —— DSv4 不该依赖另一个项目的个人目录。
#    Hy3 那边继续用它自己那份,两边各自演进。
#
# 读 prepare_dsv4_data.py 的产出,按 get_batch 的约定发批:
#   tokens (B,L) int64 | labels (B,L) 左移一位 | loss_mask (B,L) float 左移一位 |
#   position_ids (B,L)。attention_mask 不发(skip_getting_attention_mask_from_dataset=True,
#   由 TE 自己建因果掩码)。
#
# ⚠️ 不打包模式(_BinDataset)把**每一条样本都补齐到完整窗口** —— 一条 2189 token 的
#    样本也按满窗口算一次前向。原因:EP>1 时 MoE 的 all-to-all 跨 DP rank,
#    各 rank 的微批形状必须一致。代价见文档 §6.4(这是显存被推过线的放大器)。
#    长窗口请用打包模式(packed=True),它也是 CP>1 的硬前提。
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider

# DSv4 的 pad/eos 是 id 1。填充只出现在序列尾部,加上因果掩码和 loss_mask=0,
# 理论上填错 id 也不影响结果 —— 但换成词表更小的模型就会越界,别依赖这个巧合。
PAD_ID = int(os.environ.get("SFT_PAD_ID", 1))


def _apply_mask_mode(mask: np.ndarray, mode: str) -> np.ndarray:
    """Return the supervision mask selected for an ablation run.

    ``source`` is the production path: use the mask emitted by
    ``prepare_dsv4_data.py``.  ``all_nonpad`` is intentionally a diagnostic
    ablation only; it supervises every real token and is not a recommended
    SFT objective.
    """
    mode = mode.strip().lower()
    if mode == "source":
        return mask
    if mode == "all_nonpad":
        return np.ones(mask.shape, dtype=np.float32)
    raise ValueError(
        f"unsupported SFT_MASK_MODE={mode!r}; expected source or all_nonpad"
    )


class _BinDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        target_length: int,
        seq_length: int,
        seed: int = 42,
        max_samples: int = 0,
        mask_mode: str = "source",
    ):
        idx = np.load(os.path.join(data_dir, f"{split}.idx.npy"))  # (N,2) offset,len
        fit = idx[:, 1] <= seq_length
        n_skip = int((~fit).sum())
        if n_skip:
            print(f"[dsv4_sft_data] {split}: 跳过 {n_skip}/{len(idx)} 条 > {seq_length} token 的样本", flush=True)
        self._idx = idx[fit]
        if max_samples > 0:
            self._idx = self._idx[:max_samples]
        self._n = int(self._idx.shape[0])
        assert self._n > 0, f"split '{split}' 里没有一条样本装得进 seq_length={seq_length}"
        self._ids = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.int32, mode="r")
        self._mask = np.memmap(os.path.join(data_dir, f"{split}.mask.bin"), dtype=np.uint8, mode="r")
        self._L = seq_length
        order = np.arange(self._n)
        np.random.default_rng(seed).shuffle(order)
        self._order = order
        self._length = int(max(0, target_length))
        self._mask_mode = mask_mode
        self.collate_fn = self._collate

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        j = int(self._order[i % self._n])
        off, ln = int(self._idx[j, 0]), int(self._idx[j, 1])
        ids = np.asarray(self._ids[off : off + ln], dtype=np.int64)
        msk = _apply_mask_mode(
            np.asarray(self._mask[off : off + ln], dtype=np.float32),
            self._mask_mode,
        )
        return ids, msk

    def _collate(self, batch, *_):
        B, L = len(batch), self._L
        tokens = torch.full((B, L), PAD_ID, dtype=torch.long)
        mask0 = torch.zeros((B, L), dtype=torch.float)
        for i, (ids, msk) in enumerate(batch):
            n = ids.shape[0]
            tokens[i, :n] = torch.from_numpy(ids)
            mask0[i, :n] = torch.from_numpy(msk)
        # 下一个 token 作为目标:labels[t] = tokens[t+1];最后一列是填充(mask 0)。
        # 被掩掉的位置填一个**合法的** token id(0)—— mcore/TE 的融合交叉熵没有
        # -100 这种"忽略"约定,只认 loss_mask。
        labels = torch.cat([tokens[:, 1:], torch.full((B, 1), PAD_ID, dtype=torch.long)], dim=1)
        loss_mask = torch.cat([mask0[:, 1:], torch.zeros((B, 1), dtype=torch.float)], dim=1)
        labels = labels.masked_fill(loss_mask == 0, 0)
        position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, L).contiguous()
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }


class _PackedBinDataset(torch.utils.data.Dataset):
    """打包模式:一个窗口里首尾相接放几条样本 + 尾部一段填充。

    读 pack_dsv4_windows.py 的窗口分配表。发的是 gpt_step 的打包约定:
    cu_seqlens(定宽,-1 结尾)+ cu_seqlens_argmin + max_seqlen,
    position_ids 每段从 0 重数,labels 在窗口内左移一位、段边界那一位强制踢出 loss。

    ⛔ CP>1 时必须用这个 —— 内核强制 qkv_format='thd',而 thd 要 cu_seqlens。
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        target_length: int,
        seq_length: int,
        seed: int = 42,
        max_windows: int = 0,
        mask_mode: str = "source",
    ):
        import json

        with open(os.path.join(data_dir, f"{split}.windows_meta.json")) as f:
            meta = json.load(f)
        assert meta["window"] == seq_length, (
            f"打包窗口是 {meta['window']},本次 SEQ_LEN={seq_length} —— 对不上。"
            f"重打包:python3 pack_dsv4_windows.py --data-dir {data_dir} --window {seq_length}"
        )
        self._idx = np.load(os.path.join(data_dir, f"{split}.idx.npy"))
        self._offs = np.load(os.path.join(data_dir, f"{split}.windows_offsets.npy"))
        self._samples = np.load(os.path.join(data_dir, f"{split}.windows_samples.npy"))
        if max_windows > 0:
            n_windows = min(max_windows, len(self._offs) - 1)
            self._offs = self._offs[: n_windows + 1]
            self._samples = self._samples[: int(self._offs[-1])]
        self._ids = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.int32, mode="r")
        self._mask = np.memmap(os.path.join(data_dir, f"{split}.mask.bin"), dtype=np.uint8, mode="r")
        self._L = seq_length
        # 放得下每个真实段 + 尾部填充段 + 一个 -1 结束符
        self._cu_width = meta["max_segments"] + 3
        self._n = int(len(self._offs) - 1)
        order = np.arange(self._n)
        np.random.default_rng(seed).shuffle(order)
        self._order = order
        self._length = int(max(0, target_length))
        self._mask_mode = mask_mode
        self.collate_fn = self._collate

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        w = int(self._order[i % self._n])
        segs = []
        for j in self._samples[self._offs[w] : self._offs[w + 1]]:
            off, ln = int(self._idx[j, 0]), int(self._idx[j, 1])
            segs.append(
                (
                    np.asarray(self._ids[off : off + ln], dtype=np.int64),
                    _apply_mask_mode(
                        np.asarray(self._mask[off : off + ln], dtype=np.float32),
                        self._mask_mode,
                    ),
                )
            )
        return segs

    def _collate(self, batch, *_):
        B, L = len(batch), self._L
        assert B == 1, f"打包模式要 micro_batch_size=1(一个微批一个窗口),现在是 {B}"
        tokens = torch.full((B, L), PAD_ID, dtype=torch.long)
        mask0 = torch.zeros((B, L), dtype=torch.float)
        position_ids = torch.zeros((B, L), dtype=torch.long)
        cu = torch.full((B, self._cu_width), -1, dtype=torch.int32)
        cu_argmin = torch.zeros((B,), dtype=torch.long)
        max_seqlen = torch.zeros((B,), dtype=torch.int32)
        for b, segs in enumerate(batch):
            bounds = [0]
            t = 0
            for ids, msk in segs:
                n = ids.shape[0]
                tokens[b, t : t + n] = torch.from_numpy(ids)
                mask0[b, t : t + n] = torch.from_numpy(msk)
                position_ids[b, t : t + n] = torch.arange(n, dtype=torch.long)
                t += n
                bounds.append(t)
            if t < L:  # 尾部填充段(tokens 保持 PAD_ID,mask 0)
                position_ids[b, t:L] = torch.arange(L - t, dtype=torch.long)
                bounds.append(L)
            cu[b, : len(bounds)] = torch.tensor(bounds, dtype=torch.int32)
            cu_argmin[b] = len(bounds)
            max_seqlen[b] = int(np.diff(bounds).max())
        labels = torch.cat([tokens[:, 1:], torch.full((B, 1), PAD_ID, dtype=torch.long)], dim=1)
        loss_mask = torch.cat([mask0[:, 1:], torch.zeros((B, 1), dtype=torch.float)], dim=1)
        # 跨段边界的预测永远不进 loss(和样本起始掩码重复,但让它成为结构性保证)
        for b, segs in enumerate(batch):
            ends = torch.tensor(np.cumsum([s[0].shape[0] for s in segs]), dtype=torch.long)
            loss_mask[b, ends - 1] = 0.0
        labels = labels.masked_fill(loss_mask == 0, 0)
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "cu_seqlens": cu,
            "cu_seqlens_argmin": cu_argmin,
            "max_seqlen": max_seqlen,
        }


@dataclass(kw_only=True)
class DSv4PreTokenizedProvider(DatasetProvider):
    """prepare_dsv4_data.py 产出的数据提供器。"""

    seq_length: int
    data_dir: str
    seed: int = 42
    packed: bool = False
    mask_mode: str = "source"
    max_samples: int = 0
    max_windows: int = 0
    skip_getting_attention_mask_from_dataset: bool = True
    pack_sequences_in_batch: bool = False
    dataloader_type: Optional[str] = "single"

    def _make(self, split: str, samples: Optional[int]):
        if not samples or samples <= 0:
            return None
        if not os.path.exists(os.path.join(self.data_dir, f"{split}.idx.npy")):
            return None
        if self.packed:
            return _PackedBinDataset(
                self.data_dir,
                split,
                samples,
                self.seq_length,
                self.seed,
                max_windows=self.max_windows,
                mask_mode=self.mask_mode,
            )
        return _BinDataset(
            self.data_dir,
            split,
            samples,
            self.seq_length,
            self.seed,
            max_samples=self.max_samples,
            mask_mode=self.mask_mode,
        )

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        return self._make("train", context.train_samples), self._make("val", context.valid_samples), None
