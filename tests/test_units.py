"""Fast checks of the pieces that do not need vLLM:  pytest -q tests"""
import math

import pytest
import torch

from rlprof.config import load_config
from rlprof.cupti_profiler import _union_ns, categorize
from rlprof.data import last_boxed
from rlprof.losses import policy_loss
from rlprof.metrics import group_advantages, mismatch_stats, prob_hist2d
from rlprof.policy import Policy
from rlprof.quant import int8_items, quantize_int8_per_channel
from rlprof.rewards import countdown_reward, math_reward


def test_rewards():
    assert math_reward("so \\boxed{72}", "72") == 1.0
    assert math_reward("so \\boxed{1,000}", "1000") == 1.0
    assert math_reward("the answer is 18.", "18") == 1.0
    assert math_reward("\\boxed{\\frac{1}{2}}", "0.5") == 1.0
    assert math_reward("\\boxed{\\dfrac{2}{3}}", "\\frac{2}{3}") == 1.0
    assert math_reward("\\boxed{17}", "18") == 0.0
    assert last_boxed("a \\boxed{x^{2}} b \\boxed{\\frac{a}{b}}") == "\\frac{a}{b}"
    assert countdown_reward("<answer> (64 - 19) + 9 </answer>", [64, 19, 9], 54) == 1.0
    assert countdown_reward("<answer> 64 - 19 </answer>", [64, 19, 9], 45) == 0.0        # 9 unused
    assert countdown_reward("<answer> 6 / 4 * 2 </answer>", [6, 4, 2], 3) == 1.0          # exact fractions
    assert countdown_reward("<answer> __import__('os') </answer>", [1], 1) == 0.0


def test_losses_reduce_to_ppo_when_engines_agree():
    torch.manual_seed(0)
    B, T = 4, 7
    lp_old = -torch.rand(B, T)
    lp = (lp_old + 0.05 * torch.randn(B, T)).requires_grad_()
    adv, mask = torch.randn(B), (torch.rand(B, T) > 0.2).float()
    ref, _ = policy_loss(lp, lp_old, lp_old, adv, mask, "none")
    for c in ("tis", "ppo_is", "vanilla_is"):
        l, st = policy_loss(lp, lp_old, lp_old, adv, mask, c)
        assert torch.allclose(l, ref, atol=1e-6), c
        assert st["is_weight_mean"] == pytest.approx(1.0)


def test_tis_truncates():
    lp_old = torch.zeros(1, 3)
    lp_roll = torch.log(torch.tensor([[0.1, 1.0, 1.0]]))        # ratio 10, 1, 1
    lp = lp_old.clone().requires_grad_()
    _, st = policy_loss(lp, lp_old, lp_roll, torch.ones(1), torch.ones(1, 3), "tis", tis_cap=2.0)
    assert st["is_weight_max"] == pytest.approx(2.0)
    assert st["tis_truncated_frac"] == pytest.approx(1 / 3)
    _, st = policy_loss(lp, lp_old, lp_roll, torch.ones(1), torch.ones(1, 3), "vanilla_is")
    assert st["is_weight_max"] == pytest.approx(10.0, rel=1e-5)


def test_mismatch_stats():
    lr = torch.log(torch.tensor([[0.5, 0.9, 0.2]]))
    m = torch.ones(1, 3)
    s = mismatch_stats(lr, lr.clone(), m)
    assert s["kl_k3"] == pytest.approx(0.0, abs=1e-12) and s["prob_absdiff_max"] == 0
    s = mismatch_stats(lr, torch.log(torch.tensor([[0.25, 0.9, 0.2]])), m)
    assert s["prob_absdiff_max"] == pytest.approx(0.25) and s["kl_k3"] > 0
    h = prob_hist2d(lr, lr, m, bins=10)
    assert h.sum() == 3 and (h == h.T).all()


def test_group_advantages():
    a = group_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 4)
    assert torch.allclose(a[:4].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(a[4:], torch.zeros(4))


def test_int8_quant_roundtrip():
    w = torch.randn(64, 32, dtype=torch.bfloat16)
    q, s = quantize_int8_per_channel(w)
    assert q.dtype == torch.int8 and s.shape == (64, 1)
    err = (q.float() * s.float() - w.float()).abs().max() / w.float().abs().max()
    assert err < 0.01
    names = [n for n, _ in int8_items([("model.layers.0.mlp.up_proj.weight", w), ("model.norm.weight", w[0])])]
    assert names == ["model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.up_proj.weight_scale", "model.norm.weight"]


def test_microbatches_respect_budget():
    lens = [5, 100, 30, 30, 70, 1, 64]
    mbs = Policy.make_microbatches(lens, 128)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(lens)))
    for mb in mbs:
        assert len(mb) == 1 or max(lens[i] for i in mb) * len(mb) <= 128


def test_categories_and_union():
    assert categorize("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn") == "gemm"
    assert categorize("flash_fwd_splitkv_kernel") == "attention"
    # real names recorded by CUPTI: vLLM's Cutlass INT8 GEMM and the Marlin FP8 GEMM
    assert categorize("_ZN7cutlass7Kernel2I19enable_sm80_to_sm89INS_4gemm6kernel22DefaultGemmWithVisitorIaNS_6layout8RowMajorE") == "gemm_quant"
    assert categorize("_ZN6marlin6MarlinILl1125899906909960ELl2814749767172868EEEvPK4int4") == "gemm_quant"
    assert categorize("_ZN4vllm32dynamic_scaled_int8_quant_kernelIN3c108BFloat16EfEEvPKT_PaPT0_i") == "quantize"
    assert categorize("void at::native::multi_tensor_apply_kernel<...FusedAdamMathFunctor...>") == "optimizer"
    assert _union_ns([(0, 10), (5, 20), (30, 40)]) == 30


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("steps: 7\ncorrection: none\n")
    c = load_config(str(p), ["tis_cap=3", "trace_steps=1,5", "vllm_sleep=true", "eval_dataset=math"])
    assert (c.steps, c.correction, c.tis_cap, c.trace_steps, c.vllm_sleep, c.eval_dataset) == (7, "none", 3.0, [1, 5], True, "math")
    with pytest.raises(KeyError):
        load_config(None, ["not_a_key=1"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cupti_phase_attribution():
    from rlprof.cupti_profiler import CuptiProfiler
    prof = CuptiProfiler()
    prof.start()
    x = torch.randn(1024, 1024, device="cuda")
    with prof.phase("a"):
        y = x @ x
        with prof.phase("b"):
            y.cpu()
    prof.stop()
    s = prof.summarize()["phases"]
    assert s["a"]["kernels"] >= 1 and "DTOH" in s["a/b"]["memcpy"]
    assert s["a"]["kernels_attributed_by_correlation"] == s["a"]["kernels"]


def test_entropy_path_matches_log_softmax():
    from rlprof.policy import _chunk_logprob, _chunk_logprob_entropy
    torch.manual_seed(0)
    h, w, t = torch.randn(5, 8), torch.randn(11, 8), torch.randint(0, 11, (5,))
    lp, ent = _chunk_logprob_entropy(h, w, t, 1 / 0.7, False)
    z = (h @ w.T) / 0.7
    ref = z.log_softmax(-1)
    assert torch.allclose(lp, ref.gather(-1, t[:, None]).squeeze(-1), atol=1e-5)
    assert torch.allclose(ent, -(ref.exp() * ref).sum(-1), atol=1e-5)
    assert torch.allclose(_chunk_logprob(h, w, t, 1 / 0.7, False), lp, atol=1e-5)


def test_override_keeps_string_none():
    from rlprof.config import load_config
    cfg = load_config("configs/gsm8k_qwen2.5-0.5b.yaml", ["correction=none", "steps=3", "lr=2e-6"])
    assert cfg.correction == "none" and cfg.steps == 3 and cfg.lr == 2e-6
