#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive experiment loop: latent-space steganography feasibility on Stable Diffusion
====================================================================================
One file. Installs its own libraries. Runs on Google Colab (GPU runtime, T4 is enough).

How to run on Colab
    1) Runtime > Change runtime type > T4 GPU
    2) Upload this file, then in a cell:   !python stego_latent_loop.py
       (or paste the whole file into one cell and run it)
    3) Change MODE below: "quick" (smoke test, roughly 20-40 min on a T4) or "full".

What it does (the loop decides the next experiment from each result)
    E0  sanity            : load SD 1.5, generate, VAE round-trip, timing
    E1  inversion fidelity: how many sign-bits of z_T survive generate -> 8-bit image -> DDIM inversion
                            (grid over steps / guidance; self-loops with a wider grid if too lossy; stops if hopeless)
    E2  recoverability map: accuracy by position / channel / |z| / texture, JPEG drop, split-half reliability
    E3  block study       : additive embedding z' = z + alpha*s on one block at a time -> visual cost (PSNR/SSIM vs the
                            no-message image of the SAME seed) and bit accuracy. Tests whether a block ranking learned
                            on some prompts (static map) or per-prompt (oracle) beats random blocks.
                            -> if neither helps, the loop jumps straight to the report with a negative verdict
    E4  strength search   : raises alpha until accuracy target is met or PSNR floor is hit; reports capacity (bpp)
    E5  detectability     : latent-norm z-score of the stego latent, then tests a norm-preserving fix
    E6  report            : results.json, report.md and PNG plots

Everything is resumable: state is saved after every experiment (set USE_DRIVE=True to keep it on Google Drive).
Note: this is a FEASIBILITY STUDY of your LLSM idea (is a position-dependent suitability signal present?).
It does not train the LLSM network itself.
"""
import os, sys, io, json, time, math, random, importlib.util, subprocess, traceback
from types import SimpleNamespace

# ----------------------------------------------------------------------------- user settings
MODE      = os.environ.get("STEGO_MODE", "quick")      # "quick" | "full"
USE_DRIVE = False                                      # True: mount Drive and keep results there
RESUME    = True                                       # continue from saved state if it exists
MOCK      = os.environ.get("STEGO_MOCK") == "1"        # CPU dry-run with a fake model (control-flow test only)
OUT_DIR   = os.environ.get("STEGO_OUT",
                           "/content/stego_runs/run1" if os.path.isdir("/content") else "./stego_runs/run1")


# ----------------------------------------------------------------------------- libraries
def _ensure_libs():
    need = {"torch": "torch", "numpy": "numpy", "PIL": "pillow", "skimage": "scikit-image",
            "matplotlib": "matplotlib", "scipy": "scipy"}
    if not MOCK:
        need.update({"diffusers": "diffusers", "transformers": "transformers",
                     "accelerate": "accelerate", "safetensors": "safetensors"})
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("[setup] installing:", missing, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


_ensure_libs()

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr, structural_similarity as sk_ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if USE_DRIVE and not MOCK:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = "/content/drive/MyDrive/stego_runs/run1"

# ----------------------------------------------------------------------------- configuration
PRESETS = {
    "quick": dict(res=384, bs=4, n_e1=3, n_e2=10, G=4, n_e3=6, n_e4=3, steps_grid=[20], g_grid=[1.0, 3.0, 7.5]),
    "full":  dict(res=512, bs=4, n_e1=6, n_e2=32, G=4, n_e3=16, n_e4=8, steps_grid=[20, 30], g_grid=[1.0, 3.0, 7.5]),
    "mock":  dict(res=64, bs=8, n_e1=2, n_e2=4, G=4, n_e3=4, n_e4=2, steps_grid=[4], g_grid=[1.0, 3.0]),
}


def build_cfg():
    cfg = dict(
        model_ids=["stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5",
                   "CompVis/stable-diffusion-v1-4"],
        steps=20, g_gen=1.0, g_inv=1.0, jpeg_q=90, eval_jpeg=False,
        alpha=0.5, alpha_grid=[0.25, 0.5, 0.75, 1.0, 1.5], sel_frac=0.25, n_rand_sets=2, n_test_eval=3,
        target_acc=0.95, min_psnr=28.0,
        e1_pass=0.90, e1_floor=0.75, e2_reliab=0.5, mag_strong=0.05, jpeg_drop_flag=0.05,
        gain_acc=0.01, gain_psnr=0.5, norm_z_detect=3.0, max_gen_sec=45.0, seed=0,
    )
    cfg.update(PRESETS["mock" if MOCK else MODE])
    return cfg


PROMPTS = [
    "a misty mountain lake at sunrise", "a close-up portrait of an old fisherman", "a busy street market in a city",
    "a red vintage car parked near the sea", "a bowl of fresh fruit on a wooden table", "a snowy forest cabin at night",
    "an astronaut walking on a desert planet", "a cat sleeping on a sunny window sill", "a modern glass skyscraper at dusk",
    "a field of sunflowers under a blue sky", "a small boat on a calm harbor", "a cozy library with warm lamps",
    "a waterfall in a tropical jungle", "a plate of pasta with basil", "a lighthouse on a stormy coast",
    "a child flying a kite on a beach", "a steam train crossing a stone bridge", "a quiet temple in autumn leaves",
    "a futuristic city with flying vehicles", "a horse running in a green meadow", "a snowy mountain peak at dawn",
    "a street cafe in paris in the rain", "a colorful hot air balloon festival", "a rusty bicycle against a brick wall",
    "a desert canyon with golden light", "a robot repairing a clock", "a bridge over a foggy river",
    "a bakery counter with fresh bread", "a wooden pier at sunset", "a tiger resting in tall grass",
    "a rooftop garden in a big city", "a violin on an old wooden chair", "a village in the alps in summer",
    "an ancient ruin covered with moss", "a fishing village at low tide", "a night sky full of stars over a tent",
    "a vintage typewriter on a desk", "a flower shop on a sunny morning", "a canal with colorful houses",
    "an elephant crossing a river",
]


# ----------------------------------------------------------------------------- tiny helpers
def log(*a):
    msg = time.strftime("[%H:%M:%S] ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(os.path.join(OUT_DIR, "log.txt"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def zscore(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-9 else 1.0)


def pearson(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(q))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def psnr(a, b):
    v = sk_psnr(a, b, data_range=255)
    return float(min(v, 100.0))


def ssim(a, b):
    return float(sk_ssim(a, b, channel_axis=2, data_range=255))


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------- mock model (control-flow test only)
class _MockUNet:
    def __call__(self, x, t, encoder_hidden_states=None):
        return SimpleNamespace(sample=0.05 * x)

    def eval(self):
        return self


class _MockVAE:
    config = SimpleNamespace(scaling_factor=1.0)

    def decode(self, z):
        img = torch.tanh(F.interpolate(z[:, :3], scale_factor=8, mode="nearest") * 0.5)
        return SimpleNamespace(sample=img)

    def encode(self, x):
        p = F.avg_pool2d(x, 8)
        lat = torch.cat([p, p.mean(1, keepdim=True)], 1) * 2.0
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=lat))

    def eval(self):
        return self


class _MockTok:
    model_max_length = 77

    def __call__(self, text, **kw):
        return SimpleNamespace(input_ids=torch.zeros(1, 77, dtype=torch.long))


class _MockTE:
    def __call__(self, ids):
        return (torch.zeros(1, 77, 8),)

    def eval(self):
        return self


class MockPipe:
    def __init__(self):
        self.unet, self.vae, self.text_encoder, self.tokenizer = _MockUNet(), _MockVAE(), _MockTE(), _MockTok()
        betas = torch.linspace(1e-4, 2e-2, 1000)
        self.scheduler = SimpleNamespace(alphas_cumprod=torch.cumprod(1 - betas, 0),
                                         config=SimpleNamespace(num_train_timesteps=1000))


# ----------------------------------------------------------------------------- engine: SD + own DDIM + inversion
class Engine:
    def __init__(self, cfg):
        self.cfg, self.bs, self.L = cfg, cfg["bs"], cfg["res"] // 8
        if MOCK:
            pipe, self.device, self.dtype = MockPipe(), "cpu", torch.float32
        else:
            if not torch.cuda.is_available():
                raise SystemExit("No GPU found. Colab: Runtime > Change runtime type > T4 GPU, then run again.")
            from diffusers import StableDiffusionPipeline
            self.device, self.dtype, pipe, err = "cuda", torch.float16, None, None
            for mid in cfg["model_ids"]:
                try:
                    pipe = StableDiffusionPipeline.from_pretrained(mid, torch_dtype=self.dtype, safety_checker=None)
                    log("loaded", mid)
                    break
                except Exception as e:  # try next id
                    err = e
                    log("could not load", mid, "->", repr(e)[:120])
            if pipe is None:
                raise SystemExit(f"Could not download any model: {err}")
            pipe = pipe.to(self.device)
            try:
                pipe.vae.enable_slicing()
            except Exception:
                pass
        self.unet, self.vae = pipe.unet.eval(), pipe.vae.eval()
        self.te, self.tok = pipe.text_encoder.eval(), pipe.tokenizer
        self.sf = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        self.ac = pipe.scheduler.alphas_cumprod.to(self.device).float()
        self.T = int(pipe.scheduler.config.num_train_timesteps)
        self._emb = {}
        self.vae_bs = 2

    # ---- text / timesteps
    @torch.no_grad()
    def emb(self, prompt):
        if prompt not in self._emb:
            ids = self.tok(prompt, padding="max_length", max_length=self.tok.model_max_length,
                           truncation=True, return_tensors="pt").input_ids.to(self.device)
            self._emb[prompt] = self.te(ids)[0].to(self.dtype)
        return self._emb[prompt]

    def ts(self, steps):
        return np.linspace(0, self.T - 1, steps + 1).round().astype(int)[1:]

    @torch.no_grad()
    def _eps(self, x, t, cond, unc, g):
        B = x.shape[0]
        tt = torch.full((B,), int(t), device=x.device, dtype=torch.long)
        xin, c = x.to(self.dtype), cond.expand(B, -1, -1)
        if g == 1.0:
            return self.unet(xin, tt, encoder_hidden_states=c).sample.float()
        u = unc.expand(B, -1, -1)
        out = self.unet(torch.cat([xin, xin]), torch.cat([tt, tt]),
                        encoder_hidden_states=torch.cat([u, c])).sample.float()
        eu, ec = out.chunk(2)
        return eu + g * (ec - eu)

    @torch.no_grad()
    def _gen(self, z, cond, unc, steps, g):
        ts, x = self.ts(steps), z.float()
        for i in range(len(ts) - 1, -1, -1):
            a_t = self.ac[ts[i]]
            a_n = self.ac[ts[i - 1]] if i > 0 else self.ac[0]
            eps = self._eps(x, ts[i], cond, unc, g)
            x0 = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x = torch.sqrt(a_n) * x0 + torch.sqrt(1 - a_n) * eps
        return x

    @torch.no_grad()
    def _inv(self, x0, cond, unc, steps, g):
        ts, x = self.ts(steps), x0.float()
        for i in range(len(ts)):
            a_n = self.ac[ts[i]]
            t_c = ts[i - 1] if i > 0 else 0
            a_c = self.ac[t_c]
            eps = self._eps(x, t_c, cond, unc, g)
            x0p = (x - torch.sqrt(1 - a_c) * eps) / torch.sqrt(a_c)
            x = torch.sqrt(a_n) * x0p + torch.sqrt(1 - a_n) * eps
        return x

    # ---- batched public API
    @torch.no_grad()
    def generate(self, z, prompt, steps, g):
        c, u = self.emb(prompt), self.emb("")
        return torch.cat([self._gen(z[i:i + self.bs], c, u, steps, g) for i in range(0, z.shape[0], self.bs)])

    @torch.no_grad()
    def invert(self, x0, prompt, steps, g):
        c, u = self.emb(prompt), self.emb("")
        return torch.cat([self._inv(x0[i:i + self.bs], c, u, steps, g) for i in range(0, x0.shape[0], self.bs)])

    @torch.no_grad()
    def decode(self, x0):
        outs = []
        for i in range(0, x0.shape[0], self.vae_bs):
            im = self.vae.decode((x0[i:i + self.vae_bs] / self.sf).to(self.dtype)).sample.float()
            outs.append(((im / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(outs)

    @torch.no_grad()
    def encode(self, imgs):
        outs = []
        for i in range(0, len(imgs), self.vae_bs):
            t = torch.from_numpy(np.ascontiguousarray(imgs[i:i + self.vae_bs])).to(self.device)
            t = t.permute(0, 3, 1, 2).float() / 127.5 - 1
            outs.append(self.vae.encode(t.to(self.dtype)).latent_dist.mean.float() * self.sf)
        return torch.cat(outs)

    @torch.no_grad()
    def channel(self, z, prompt, steps, g, g_inv, jpeg_qs=(None,)):
        """z -> SD image (8-bit) -> optional JPEG -> VAE encode -> DDIM inversion. Returns clean imgs, [z_hat per attack]."""
        imgs = self.decode(self.generate(z, prompt, steps, g))
        zrs = []
        for q in jpeg_qs:
            im = imgs if q is None else np.stack([jpeg(x, q) for x in imgs])
            zrs.append(self.invert(self.encode(im), prompt, steps, g_inv))
        return imgs, zrs


def make_z(eng, seed, n=1):
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randn((n, 4, eng.L, eng.L), generator=gen).to(eng.device)


def rand_sign(eng, seed):
    gen = torch.Generator().manual_seed(int(seed))
    return ((torch.randn((1, 4, eng.L, eng.L), generator=gen) > 0).float() * 2 - 1).to(eng.device)


def block_mask(eng, G, k):
    m, b = torch.zeros(1, 1, eng.L, eng.L), eng.L // G
    r, c = divmod(int(k), G)
    m[..., r * b:(r + 1) * b, c * b:(c + 1) * b] = 1
    return m.to(eng.device)


def union_mask(eng, G, blocks):
    m = torch.zeros(1, 1, eng.L, eng.L, device=eng.device)
    for k in blocks:
        m = torch.clamp(m + block_mask(eng, G, k), max=1.0)
    return m


def sign_acc(z, zr):
    return float((torch.sign(z) == torch.sign(zr)).float().mean().item())


def texture_blocks(img, L, G):
    gray = img.astype(np.float32).mean(2)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    f = img.shape[0] // L
    m = mag[:L * f, :L * f].reshape(L, f, L, f).mean((1, 3))
    b = L // G
    return m[:G * b, :G * b].reshape(G, b, G, b).mean((1, 3))


# ----------------------------------------------------------------------------- shared evaluation of a block set
def eval_sets(ctx, p, sets, alpha, fix_norm=False):
    """Embed random +-alpha in the union of each block set (additive, shared seed), run the channel,
    return per-set dict(acc, acc_jpeg, psnr, ssim, nz). PSNR/SSIM reference = same-seed image WITHOUT message."""
    eng, cfg = ctx.eng, ctx.cfg
    G, prompt = cfg["G"], PROMPTS[p % len(PROMPTS)]
    z = make_z(eng, 1000 + p)
    D = float(z.numel())
    zs, meta = [z], []
    for j, bl in enumerate(sets):
        m, s = union_mask(eng, G, bl), rand_sign(eng, 9000 + p * 131 + j)
        zp = z + alpha * s * m
        if fix_norm:
            zp = zp * (z.norm() / zp.norm())
        zs.append(zp)
        meta.append((m, s))
    qs = (None, cfg["jpeg_q"]) if cfg["eval_jpeg"] else (None,)
    imgs, zrs = eng.channel(torch.cat(zs), prompt, cfg["steps"], cfg["g_gen"], cfg["g_inv"], qs)
    out = []
    for j, (m, s) in enumerate(meta):
        r = dict(psnr=psnr(imgs[0], imgs[j + 1]), ssim=ssim(imgs[0], imgs[j + 1]))
        cnt = float(m.sum().item() * 4)
        for qi, key in enumerate(("acc", "acc_jpeg")[:len(qs)]):
            d = zrs[qi][j + 1:j + 2] - z
            r[key] = float((((torch.sign(d) == s) & (m > 0)).float().sum() / cnt).item())
        zp = zs[j + 1]
        r["nz"] = float((((zp ** 2).sum() - D) / math.sqrt(2 * D)).item())
        r["nz_base"] = float((((z ** 2).sum() - D) / math.sqrt(2 * D)).item())
        out.append(r)
    return out


def better(a, b, cfg):
    """a beats b if it gains accuracy without losing quality, or gains quality without losing accuracy."""
    return ((a["acc"] - b["acc"] >= cfg["gain_acc"] and a["psnr"] >= b["psnr"] - 0.3) or
            (a["psnr"] - b["psnr"] >= cfg["gain_psnr"] and a["acc"] >= b["acc"] - cfg["gain_acc"]))


# ============================================================================= EXPERIMENTS
# every experiment returns (result_dict, next_experiment_name_or_None, note)
def E0(ctx):
    eng, cfg = ctx.eng, ctx.cfg
    z = make_z(eng, 1)
    t0 = time.time()
    x0 = eng.generate(z, PROMPTS[0], cfg["steps"], 7.5 if not MOCK else 3.0)
    img = eng.decode(x0)[0]
    gen_sec = time.time() - t0
    Image.fromarray(img).save(os.path.join(OUT_DIR, "sample_E0.png"))
    rt = eng.decode(eng.encode(eng.decode(x0)))[0]
    res = dict(gen_sec=gen_sec, vae_roundtrip_psnr=psnr(img, rt), res=cfg["res"], latent=eng.L,
               capacity_ceiling_bpp=(4 * eng.L * eng.L) / float(cfg["res"] ** 2))
    note = "pipeline OK"
    if gen_sec > cfg["max_gen_sec"]:
        for k in ("n_e2", "n_e3", "n_e4"):
            cfg[k] = max(2, cfg[k] // 2)
        note = "generation is slow -> halving the number of prompts in E2/E3/E4"
    return res, "E1", note


def E1(ctx):
    eng, cfg = ctx.eng, ctx.cfg
    rows = []
    for steps in cfg["steps_grid"]:
        for g in cfg["g_grid"]:
            accs, cors, t0 = [], [], time.time()
            for p in range(cfg["n_e1"]):
                z = make_z(eng, 100 + p)
                _, (zr,) = eng.channel(z, PROMPTS[p], steps, g, cfg["g_inv"])
                accs.append(sign_acc(z, zr))
                cors.append(pearson(z.cpu().numpy(), zr.cpu().numpy()))
            rows.append(dict(steps=steps, g=g, acc=float(np.mean(accs)), corr=float(np.mean(cors)),
                             sec=(time.time() - t0) / cfg["n_e1"]))
            log(f"  E1 steps={steps} g={g}: sign-acc={rows[-1]['acc']:.3f} corr={rows[-1]['corr']:.3f}")
    best = max(rows, key=lambda r: (round(r["acc"], 2), r["g"]))
    ctx.v["e1_best"] = best
    res = dict(rows=rows, best=best)
    if best["acc"] >= cfg["e1_pass"]:
        cfg["steps"], cfg["g_gen"] = best["steps"], best["g"]
        return res, "E2", f"inversion good ({best['acc']:.3f}); using steps={best['steps']} g={best['g']}"
    if ctx.state["visits"].get("E1", 0) <= 1:
        cfg["steps_grid"], cfg["g_grid"] = [max(cfg["steps_grid"]) + 30], [1.0]
        return res, "E1", f"inversion lossy ({best['acc']:.3f}) -> retry with a longer schedule (self-loop)"
    if best["acc"] >= cfg["e1_floor"]:
        cfg["steps"], cfg["g_gen"] = best["steps"], best["g"]
        ctx.v["weak_inversion"] = True
        return res, "E2", f"weak inversion ({best['acc']:.3f}) but above floor -> continue with caution"
    ctx.v["inversion_not_viable"] = True
    return res, "E6", f"inversion too lossy ({best['acc']:.3f}) -> inversion-based extraction not viable; jumping to report"


def E2(ctx):
    eng, cfg = ctx.eng, ctx.cfg
    G, L = cfg["G"], eng.L
    b = L // G
    Zs, M0, MJ, TEX = [], [], [], []
    for p in range(cfg["n_e2"]):
        z = make_z(eng, 2000 + p)
        imgs, zrs = eng.channel(z, PROMPTS[p % len(PROMPTS)], cfg["steps"], cfg["g_gen"], cfg["g_inv"],
                                (None, cfg["jpeg_q"]))
        Zs.append(z[0].cpu().numpy())
        M0.append((torch.sign(z) == torch.sign(zrs[0]))[0].cpu().numpy())
        MJ.append((torch.sign(z) == torch.sign(zrs[1]))[0].cpu().numpy())
        TEX.append(texture_blocks(imgs[0], L, G))
        if (p + 1) % 5 == 0:
            log(f"  E2 {p + 1}/{cfg['n_e2']}")
    Zs, M0, MJ = np.stack(Zs), np.stack(M0).astype(float), np.stack(MJ).astype(float)
    blk = lambda M: M[:, :, :G * b, :G * b].reshape(len(M), 4, G, b, G, b).mean(axis=(0, 1, 3, 5))
    bmap, bmap_j = blk(M0), blk(MJ)
    half = lambda M, s: blk(M[s::2]) if len(M[s::2]) else blk(M)
    reliab = pearson(half(M0, 0), half(M0, 1))
    absz = np.abs(Zs)
    edges = np.quantile(absz, np.linspace(0, 1, 11))
    mag_acc = [float(M0[(absz >= edges[i]) & (absz <= edges[i + 1])].mean()) for i in range(10)]
    mag_gain = float(np.mean(mag_acc[-3:]) - np.mean(mag_acc[:3]))
    tex = np.mean(TEX, axis=0)
    res = dict(acc=float(M0.mean()), acc_jpeg=float(MJ.mean()), jpeg_drop=float(M0.mean() - MJ.mean()),
               block_acc=bmap, block_acc_jpeg=bmap_j, channel_acc=M0.mean(axis=(0, 2, 3)),
               mag_bins_acc=mag_acc, mag_gain=mag_gain, split_half_reliability=reliab,
               acc_vs_texture_r=pearson(bmap, tex), block_texture=tex,
               block_acc_spread=float(bmap.max() - bmap.min()))
    ctx.v["static_reliable"] = bool(reliab >= cfg["e2_reliab"])
    ctx.v["mag_strong"] = bool(mag_gain >= cfg["mag_strong"])
    if res["jpeg_drop"] > cfg["jpeg_drop_flag"]:
        cfg["eval_jpeg"] = True
    note = (f"acc={res['acc']:.3f} jpeg_drop={res['jpeg_drop']:.3f} split-half r={reliab:.2f} "
            f"mag_gain={mag_gain:.3f} -> block-level study")
    if res["acc"] < 0.6:
        ctx.v["inversion_not_viable"] = True
        return res, "E6", "recovery near chance -> report"
    return res, "E3", note


def E3(ctx):
    eng, cfg = ctx.eng, ctx.cfg
    G, alpha, P = cfg["G"], cfg["alpha"], cfg["n_e3"]
    K = G * G
    R, PS, SS, TX = np.zeros((P, K)), np.zeros((P, K)), np.zeros((P, K)), np.zeros((P, K))
    for p in range(P):
        z = make_z(eng, 1000 + p)
        zs, masks, signs = [z], [], []
        for k in range(K):
            m, s = block_mask(eng, G, k), rand_sign(eng, 7000 + p * 101 + k)
            zs.append(z + alpha * s * m)
            masks.append(m)
            signs.append(s)
        imgs, (zr,) = eng.channel(torch.cat(zs), PROMPTS[p % len(PROMPTS)], cfg["steps"], cfg["g_gen"], cfg["g_inv"])
        tb = texture_blocks(imgs[0], eng.L, G).ravel()
        for k in range(K):
            d = zr[k + 1:k + 2] - z
            R[p, k] = float((((torch.sign(d) == signs[k]) & (masks[k] > 0)).float().sum() / (masks[k].sum() * 4)).item())
            PS[p, k], SS[p, k], TX[p, k] = psnr(imgs[0], imgs[k + 1]), ssim(imgs[0], imgs[k + 1]), tb[k]
        log(f"  E3 prompt {p + 1}/{P}: mean acc={R[p].mean():.3f} mean psnr={PS[p].mean():.1f}")
    train = [p for p in range(P) if p % 2 == 0]
    test = [p for p in range(P) if p % 2 == 1][:cfg["n_test_eval"]] or [P - 1]
    S_all = np.stack([zscore(R[p]) + zscore(PS[p]) for p in range(P)])
    S_static = zscore(R[train].mean(0)) + zscore(PS[train].mean(0))
    nsel = max(1, int(round(cfg["sel_frac"] * K)))
    static_sel = [int(i) for i in np.argsort(-S_static)[:nsel]]
    rng = np.random.RandomState(cfg["seed"])
    agg = {"static": [], "oracle": [], "random": []}
    for p in test:
        oracle = [int(i) for i in np.argsort(-S_all[p])[:nsel]]
        rands = [[int(i) for i in rng.choice(K, nsel, replace=False)] for _ in range(cfg["n_rand_sets"])]
        out = eval_sets(ctx, p, [static_sel, oracle] + rands, alpha)
        agg["static"].append(out[0])
        agg["oracle"].append(out[1])
        agg["random"].extend(out[2:])
        log(f"  E3 eval prompt {p}: static acc={out[0]['acc']:.3f}/psnr={out[0]['psnr']:.1f} | "
            f"oracle {out[1]['acc']:.3f}/{out[1]['psnr']:.1f} | random {np.mean([o['acc'] for o in out[2:]]):.3f}/"
            f"{np.mean([o['psnr'] for o in out[2:]]):.1f}")
    mean = lambda lst: {k: float(np.mean([r[k] for r in lst])) for k in ("acc", "psnr", "ssim")}
    s, o, r = mean(agg["static"]), mean(agg["oracle"]), mean(agg["random"])
    static_gain, oracle_gain = better(s, r, cfg), better(o, r, cfg)
    ctx.v.update(static_sel=static_sel, S_prompt=S_all.tolist(), static_gain=bool(static_gain),
                 oracle_gain=bool(oracle_gain))
    res = dict(R=R, PSNR=PS, SSIM=SS, block_acc_mean=R.mean(0).reshape(G, G), block_psnr_mean=PS.mean(0).reshape(G, G),
               corr_acc_psnr=pearson(R.mean(0), PS.mean(0)), corr_acc_texture=pearson(R.mean(0), TX.mean(0)),
               corr_psnr_texture=pearson(PS.mean(0), TX.mean(0)), static_sel=static_sel,
               eval=dict(static=s, oracle=o, random=r), static_gain=bool(static_gain), oracle_gain=bool(oracle_gain),
               n_train=len(train), n_test=len(test))
    if static_gain:
        ctx.v["sel_mode"] = "static"
        return res, "E4", "a static block ranking beats random blocks -> strength search with that selection"
    if oracle_gain:
        ctx.v["sel_mode"] = "oracle"
        return res, "E4", "only a per-prompt (oracle) ranking helps -> map must depend on the latent; strength search with oracle blocks"
    ctx.v["sel_mode"] = None
    return res, "E6", "no selection beats random blocks at this granularity -> negative verdict, skipping E4/E5"


def E4(ctx):
    eng, cfg = ctx.eng, ctx.cfg
    G, mode = cfg["G"], ctx.v.get("sel_mode", "static")
    n = min(cfg["n_e4"], cfg["n_e3"])
    nsel = max(1, int(round(cfg["sel_frac"] * G * G)))
    carriers = nsel * 4 * (eng.L // G) ** 2
    rows, op = [], None
    for alpha in cfg["alpha_grid"]:
        outs = []
        for p in range(n):
            sel = ctx.v["static_sel"] if mode == "static" else [int(i) for i in np.argsort(-np.array(ctx.v["S_prompt"][p]))[:nsel]]
            outs.append(eval_sets(ctx, p, [sel], alpha)[0])
        row = dict(alpha=alpha, acc=float(np.mean([o["acc"] for o in outs])), psnr=float(np.mean([o["psnr"] for o in outs])),
                   ssim=float(np.mean([o["ssim"] for o in outs])))
        if cfg["eval_jpeg"]:
            row["acc_jpeg"] = float(np.mean([o["acc_jpeg"] for o in outs]))
        row["bpp_raw"] = carriers / float(cfg["res"] ** 2)
        row["bpp_success"] = row["bpp_raw"] * row["acc"]
        rows.append(row)
        log(f"  E4 alpha={alpha}: acc={row['acc']:.3f} psnr={row['psnr']:.1f} ssim={row['ssim']:.3f}")
        if row["acc"] >= cfg["target_acc"] and (not cfg["eval_jpeg"] or row["acc_jpeg"] >= cfg["target_acc"] - 0.03):
            op = row
            break
    op = op or max(rows, key=lambda r: r["acc"])
    ctx.v["psnr_floor_met"] = bool(op["psnr"] >= cfg["min_psnr"])
    cfg["alpha"] = op["alpha"]
    ctx.v["op_point"] = op
    ctx.v["target_reached"] = bool(op["acc"] >= cfg["target_acc"])
    note = (f"operating point alpha={op['alpha']} acc={op['acc']:.3f} psnr={op['psnr']:.1f} "
            f"(floor {cfg['min_psnr']} dB met: {ctx.v['psnr_floor_met']}) bpp_success={op['bpp_success']:.4f}")
    return dict(rows=rows, op=op, carriers=carriers, sel_mode=mode), "E5", note


def E5(ctx):
    eng, cfg = ctx.eng, ctx.cfg
    G, mode, alpha = cfg["G"], ctx.v.get("sel_mode", "static"), cfg["alpha"]
    n = min(cfg["n_e4"], cfg["n_e3"])
    nsel = max(1, int(round(cfg["sel_frac"] * G * G)))
    raw, fix = [], []
    for p in range(n):
        sel = ctx.v["static_sel"] if mode == "static" else [int(i) for i in np.argsort(-np.array(ctx.v["S_prompt"][p]))[:nsel]]
        raw.append(eval_sets(ctx, p, [sel], alpha, fix_norm=False)[0])
        fix.append(eval_sets(ctx, p, [sel], alpha, fix_norm=True)[0])
    mean = lambda lst, k: float(np.mean([r[k] for r in lst]))
    res = dict(alpha=alpha, norm_z_raw=mean(raw, "nz"), norm_z_base=mean(raw, "nz_base"), norm_z_fixed=mean(fix, "nz"),
               acc_raw=mean(raw, "acc"), acc_fixed=mean(fix, "acc"), psnr_raw=mean(raw, "psnr"), psnr_fixed=mean(fix, "psnr"))
    res["detectable_by_norm"] = bool(abs(res["norm_z_raw"]) > cfg["norm_z_detect"])
    res["fix_keeps_accuracy"] = bool(res["acc_fixed"] >= res["acc_raw"] - 0.02)
    ctx.v["norm"] = res
    return res, "E6", (f"norm z-score raw={res['norm_z_raw']:.2f} fixed={res['norm_z_fixed']:.2f}; "
                       f"acc raw={res['acc_raw']:.3f} fixed={res['acc_fixed']:.3f}")


def E6(ctx):
    cfg, v, d = ctx.cfg, ctx.v, ctx.state["done"]
    L = []
    L.append("# Latent steganography feasibility report\n")
    L.append(f"mode={MODE if not MOCK else 'mock'}  resolution={cfg['res']}  latent={cfg['res'] // 8}x{cfg['res'] // 8}x4")
    cap = d.get("E0", {}).get("result", {}).get("capacity_ceiling_bpp", 0.0625)
    L.append(f"\nCapacity ceiling at 1 bit per latent coordinate: **{cap:.4f} bpp** (independent of resolution).\n")
    verdict = "INCONCLUSIVE"
    if v.get("aborted"):
        verdict = "ABORTED: " + str(v["aborted"])[:200]
    elif v.get("inversion_not_viable"):
        verdict = ("NEGATIVE-INVERSION: sign bits of z_T do not survive generate->8-bit image->DDIM inversion well enough. "
                   "Use a learned blind extractor (decoder network) instead of inversion-based extraction.")
    elif v.get("sel_mode") is None and "E3" in d:
        verdict = ("NEGATIVE-MAP: neither a static nor a per-prompt block ranking beat random blocks. At this granularity "
                   "the suitability signal is weak; try finer blocks (G=8), more prompts, or a different perturbation scheme "
                   "before investing in training an LLSM.")
    elif v.get("sel_mode") == "static":
        verdict = ("POSITIVE-STATIC: a block ranking learned on some prompts transfers to unseen prompts. "
                   "A position-dependent map has value; an LLSM can start as a prompt-independent prior.")
    elif v.get("sel_mode") == "oracle":
        verdict = ("POSITIVE-DYNAMIC: suitability depends on the specific latent/prompt. A learned LLSM conditioned on z "
                   "(or on the prompt) is justified; a fixed map is not enough.")
    L.append(f"## Verdict\n**{verdict}**\n")
    if "E1" in d:
        b = d["E1"]["result"]["best"]
        L.append(f"- E1 best inversion: steps={b['steps']} g={b['g']} sign-acc={b['acc']:.3f}")
    if "E2" in d:
        r = d["E2"]["result"]
        L.append(f"- E2 accuracy={r['acc']:.3f}, JPEG drop={r['jpeg_drop']:.3f}, split-half reliability={r['split_half_reliability']:.2f}, "
                 f"|z| gain (top-3 vs bottom-3 deciles)={r['mag_gain']:.3f}, acc-vs-texture r={r['acc_vs_texture_r']:.2f}")
    if "E3" in d:
        e = d["E3"]["result"]["eval"]
        for k in ("static", "oracle", "random"):
            L.append(f"- E3 {k}: acc={e[k]['acc']:.3f} psnr={e[k]['psnr']:.2f} ssim={e[k]['ssim']:.3f}")
    if v.get("op_point"):
        o = v["op_point"]
        L.append(f"- E4 operating point: alpha={o['alpha']} acc={o['acc']:.3f} psnr={o['psnr']:.2f} ssim={o['ssim']:.3f} "
                 f"bpp_raw={o['bpp_raw']:.4f} bpp_success={o['bpp_success']:.4f} (accuracy target reached: {v.get('target_reached')}, "
                 f"PSNR floor {cfg['min_psnr']} dB met: {v.get('psnr_floor_met')})")
    if v.get("norm"):
        n = v["norm"]
        L.append(f"- E5 latent-norm z-score: raw={n['norm_z_raw']:.2f}, fixed={n['norm_z_fixed']:.2f}, "
                 f"detectable by norm test={n['detectable_by_norm']}, fix keeps accuracy={n['fix_keeps_accuracy']}")
    L.append("\nPSNR/SSIM reference = image generated from the same seed and prompt without any message.")
    L.append("Bit accuracy is raw (no error-correcting code). bpp_success = embedded coordinates x accuracy / pixels.")
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(L))
    try:
        _plots(ctx)
    except Exception as e:
        log("plot error:", repr(e)[:200])
    print("\n".join(L))
    return dict(verdict=verdict), None, "report written to " + OUT_DIR


def _plots(ctx):
    d = ctx.state["done"]
    if "E2" in d:
        r = d["E2"]["result"]
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
        ax[0].plot(range(1, 11), r["mag_bins_acc"], "o-")
        ax[0].set(title="bit accuracy vs |z| decile", xlabel="decile", ylabel="accuracy")
        im = ax[1].imshow(np.array(r["block_acc"]), cmap="viridis")
        ax[1].set_title("E2 block accuracy")
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(np.array(r["block_texture"]), cmap="magma")
        ax[2].set_title("block texture")
        plt.colorbar(im, ax=ax[2])
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "plot_E2.png"), dpi=130)
        plt.close(fig)
    if "E3" in d:
        r = d["E3"]["result"]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        im = ax[0].imshow(np.array(r["block_acc_mean"]), cmap="viridis")
        ax[0].set_title("E3 accuracy per block")
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(np.array(r["block_psnr_mean"]), cmap="plasma")
        ax[1].set_title("E3 PSNR per block (higher = cheaper)")
        plt.colorbar(im, ax=ax[1])
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "plot_E3.png"), dpi=130)
        plt.close(fig)
    if "E4" in d:
        rows = d["E4"]["result"]["rows"]
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot([r["psnr"] for r in rows], [r["acc"] for r in rows], "o-")
        for r in rows:
            ax.annotate(str(r["alpha"]), (r["psnr"], r["acc"]))
        ax.set(xlabel="PSNR (dB)", ylabel="bit accuracy", title="E4 capacity-quality-accuracy trade-off")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "plot_E4.png"), dpi=130)
        plt.close(fig)


EXPERIMENTS = {"E0": E0, "E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5, "E6": E6}


# ============================================================================= controller loop
class Ctx:
    def __init__(self, cfg):
        self.cfg, self.eng = cfg, None
        self.state = dict(cfg=cfg, v={}, done={}, next="E0", visits={})
        self.v = self.state["v"]

    def path(self):
        return os.path.join(OUT_DIR, "state.json")

    def load(self):
        if RESUME and os.path.exists(self.path()):
            self.state = json.load(open(self.path()))
            self.cfg, self.v = self.state["cfg"], self.state["v"]
            log("resuming from saved state; next =", self.state["next"])

    def save(self):
        with open(self.path(), "w") as f:
            json.dump(self.state, f, default=jdefault)

    def ensure_engine(self):
        if self.eng is None:
            self.eng = Engine(self.cfg)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    ctx = Ctx(build_cfg())
    ctx.load()
    if ctx.state["next"] is None:
        log("this run already finished; delete", OUT_DIR, "or set RESUME=False to start again")
        return ctx
    ctx.ensure_engine()
    ctx.eng.bs = ctx.cfg["bs"]
    name, hops = ctx.state["next"], 0
    while name and hops < 20:
        hops += 1
        ctx.state["visits"][name] = ctx.state["visits"].get(name, 0) + 1
        log(f"=== {name} (visit {ctx.state['visits'][name]}) ===")
        t0 = time.time()
        try:
            res, nxt, note = EXPERIMENTS[name](ctx)
        except Exception as e:
            if "out of memory" in str(e).lower() and ctx.eng.bs > 1:
                ctx.eng.bs = max(1, ctx.eng.bs // 2)
                ctx.cfg["bs"] = ctx.eng.bs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log("out of memory -> batch size reduced to", ctx.eng.bs, "and retrying", name)
                continue
            log("ERROR in", name, "\n" + traceback.format_exc())
            if name == "E6":
                break
            ctx.v["aborted"] = f"{name}: {e}"
            res, nxt, note = dict(error=str(e)), "E6", "aborted -> report"
        ctx.state["done"][name] = dict(result=res, note=note, seconds=time.time() - t0)
        ctx.state["next"] = nxt
        ctx.save()
        log(f"{name} done in {time.time() - t0:.0f}s | {note} | next -> {nxt}")
        name = nxt
    log("finished. Results:", OUT_DIR, "(report.md, state.json, plots)")
    return ctx


if __name__ == "__main__":
    main()
