"""Tests for the LLM inference analysis module."""
import pytest

from helix.inference import (
    ModelConfig, analyze_inference, print_inference_report,
    kv_footprint, plan_kv_offload, default_tiers, MemoryTier,
    available_presets,
    serving_benchmark, print_serving_result,
    sweep_inference, print_sweep, sweep_to_markdown, sweep_to_csv,
)


def cfg7b():
    return ModelConfig.from_preset("llama2-7b")


class TestModelConfig:
    def test_presets_load(self):
        for name in available_presets():
            c = ModelConfig.from_preset(name)
            assert c.num_params > 0
            assert c.d_model == c.num_heads * c.head_dim

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            ModelConfig.from_preset("does-not-exist")

    def test_llama2_7b_param_count(self):
        # llama2-7b should land close to 7 billion params.
        c = cfg7b()
        assert 6.0e9 < c.num_params < 7.5e9

    def test_gqa_shrinks_kv(self):
        mha = ModelConfig("mha", 32, 32, 32, 128, 11008)
        gqa = ModelConfig("gqa", 32, 32, 8, 128, 11008)
        assert gqa.kv_bytes_per_token() < mha.kv_bytes_per_token()
        assert gqa.kv_bytes_per_token() == mha.kv_bytes_per_token() // 4

    def test_dtype_scales_weights(self):
        fp16 = ModelConfig.from_preset("llama2-7b", dtype_bytes=2)
        fp8 = ModelConfig.from_preset("llama2-7b", dtype_bytes=1)
        assert fp16.weight_bytes == 2 * fp8.weight_bytes


class TestKVFootprint:
    def test_scales_with_batch_and_context(self):
        c = cfg7b()
        base = kv_footprint(c, batch=1, context_len=1024)
        big = kv_footprint(c, batch=4, context_len=2048)
        assert big.total_bytes == base.total_bytes * 4 * 2

    def test_total_equals_per_seq_times_batch(self):
        c = cfg7b()
        f = kv_footprint(c, batch=8, context_len=512)
        assert f.total_bytes == f.bytes_per_sequence * 8


class TestOffloadPlanner:
    def _tiers(self):
        return [
            MemoryTier("GPU HBM", 10e9, 2000e9),
            MemoryTier("CPU RAM", 100e9, 55e9),
            MemoryTier("NVMe SSD", 1e12, 6e9),
        ]

    def test_fits_in_gpu(self):
        plan = plan_kv_offload(5e9, self._tiers())
        assert plan.fits_in_gpu
        assert plan.slowdown_vs_hbm == pytest.approx(1.0)
        assert len(plan.placements) == 1

    def test_spills_to_cpu(self):
        plan = plan_kv_offload(15e9, self._tiers())
        assert not plan.fits_in_gpu
        assert plan.placements[0]["tier"] == "GPU HBM"
        assert plan.placements[1]["tier"] == "CPU RAM"
        assert plan.slowdown_vs_hbm > 1.0

    def test_all_bytes_placed(self):
        total = 15e9
        plan = plan_kv_offload(total, self._tiers())
        assert sum(p["bytes"] for p in plan.placements) == pytest.approx(total)

    def test_default_tiers_gpu_first_and_fastest(self):
        # Hierarchy is locality-ordered (GPU→CPU→SSD→remote); GPU HBM leads and
        # has the highest bandwidth of the set.
        tiers = default_tiers("A100", gpu_reserve_bytes=20e9)
        assert tiers[0].name == "GPU HBM"
        assert tiers[0].bandwidth_bytes_s == max(t.bandwidth_bytes_s for t in tiers)


class TestAnalyzeInference:
    def test_returns_expected_keys(self):
        r = analyze_inference(cfg7b(), batch=1, prompt_len=512, gen_len=64,
                              device="A100")
        for key in ("config", "kv", "offload", "prefill", "decode",
                    "ttft_ms", "ms_per_output_token", "decode_tokens_per_s",
                    "total_latency_ms", "advice"):
            assert key in r

    def test_prefill_compute_decode_memory(self):
        # The central claim: prefill is compute-bound, decode is memory-bound.
        r = analyze_inference(cfg7b(), batch=1, prompt_len=2048, gen_len=128,
                              device="A100")
        assert r["prefill"]["is_compute_bound"] is True
        assert r["decode"]["is_compute_bound"] is False

    def test_decode_lower_intensity_than_prefill(self):
        r = analyze_inference(cfg7b(), batch=8, prompt_len=1024, gen_len=128,
                              device="H100")
        assert (r["decode"]["arithmetic_intensity"]
                < r["prefill"]["arithmetic_intensity"])

    def test_latencies_positive(self):
        r = analyze_inference(cfg7b(), batch=1, prompt_len=512, gen_len=32,
                              device="A100")
        assert r["ttft_ms"] > 0
        assert r["ms_per_output_token"] > 0
        assert r["decode_tokens_per_s"] > 0

    def test_large_batch_spills_to_offload(self):
        # A huge batch of long contexts must exceed 80 GB of HBM.
        r = analyze_inference(cfg7b(), batch=256, prompt_len=4096, gen_len=256,
                              device="A100")
        assert r["offload"]["fits_in_gpu"] is False
        assert r["offload"]["slowdown_vs_hbm"] > 1.0

    def test_advice_nonempty(self):
        r = analyze_inference(cfg7b(), batch=1, prompt_len=512, gen_len=32,
                              device="A100")
        assert isinstance(r["advice"], list) and len(r["advice"]) >= 1
        assert all(isinstance(s, str) for s in r["advice"])

    def test_print_runs(self, capsys):
        r = analyze_inference(cfg7b(), batch=4, prompt_len=1024, gen_len=64,
                              device="A100")
        print_inference_report(r)
        out = capsys.readouterr().out
        assert "Inference Roofline" in out
        assert "KV cache" in out
        assert "Prefill" in out
        assert "Decode" in out


class TestServingBenchmark:
    def test_analytic_fallback_when_no_vllm(self):
        # No model / no vLLM on CI → analytic backend.
        res = serving_benchmark(cfg7b(), batch=1, prompt_len=512, gen_len=32,
                                device="A100")
        assert res.backend == "analytic"
        assert res.ttft_ms > 0
        assert res.tpot_ms > 0
        assert res.throughput_tok_s > 0

    def test_analytic_matches_analyze_inference(self):
        res = serving_benchmark(cfg7b(), batch=2, prompt_len=512, gen_len=16,
                                device="A100")
        ref = analyze_inference(cfg7b(), batch=2, prompt_len=512, gen_len=16,
                                device="A100")
        assert res.ttft_ms == ref["ttft_ms"]
        assert res.tpot_ms == ref["ms_per_output_token"]

    def test_force_vllm_without_install_raises(self):
        with pytest.raises(RuntimeError):
            serving_benchmark(cfg7b(), device="A100", model="fake/model",
                              backend="vllm")

    def test_print_serving_runs(self, capsys):
        res = serving_benchmark(cfg7b(), batch=1, prompt_len=256, gen_len=16,
                                device="A100")
        print_serving_result(res)
        out = capsys.readouterr().out
        assert "Serving Benchmark" in out
        assert "TTFT" in out


class TestSweep:
    def test_sweep_row_count(self):
        rows = sweep_inference(cfg7b(), batches=[1, 8, 32], prompt_lens=[1024, 2048],
                               gen_len=64, device="A100")
        assert len(rows) == 3 * 2

    def test_throughput_grows_with_batch_until_spill(self):
        rows = sweep_inference(cfg7b(), batches=[1, 8, 32], prompt_lens=[1024],
                               gen_len=64, device="A100")
        # While the cache fits, bigger batch => higher throughput.
        fitting = [r for r in rows if r.fits_hbm]
        tputs = [r.throughput_tok_s for r in fitting]
        assert tputs == sorted(tputs)

    def test_markdown_and_csv(self):
        rows = sweep_inference(cfg7b(), batches=[1, 8], prompt_lens=[1024],
                               gen_len=32, device="A100")
        md = sweep_to_markdown(rows)
        assert md.startswith("|") and "tok/s" in md
        csv = sweep_to_csv(rows)
        assert "batch" in csv.splitlines()[0]
        assert len(csv.splitlines()) == len(rows) + 1

    def test_print_sweep_runs(self, capsys):
        rows = sweep_inference(cfg7b(), batches=[1, 8], prompt_lens=[1024],
                               gen_len=32, device="A100")
        print_sweep(rows, cfg_name="llama2-7b", device="A100")
        out = capsys.readouterr().out
        assert "Sweep" in out and "tok/s" in out
