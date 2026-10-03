"""Split-K (flash-decoding) attention over a packed or bf16 prefix plus a current block.

One kernel family serves every attention over the cache: dense decoding, step 0 of
every selection method, the lookahead probe and chunked prefill.

    out, stats = splitk_attention(q, view, kc, vc, want_prefix=True)
    mass, score = prefix_scores(q, view, stats, q_rows, value=True)

Layout. Q is [B, Hq, T, D]; the G = Hq / Hkv query heads of one KV head and its T
positions form R = G*T rows that share every K/V tile, so a tile is read once per
KV head (not once per query head). Pass 1 splits the prefix keys over programs and
writes partial (max, sum, acc); the current block is one more split, optionally
block-causal. The combine kernel merges them into the output and, on request, the
prefix-only softmax statistics (M_p, L_p) and output O_p per row.

Pass 2 (prefix_scores) re-reads the prefix tile by tile and returns, per entry, the
prefix-softmax mass averaged over the selection rows, a_j, and a_j * ||v_j - c|| with
c the a-weighted value centre, which equals the mean of O_p over the selection rows.

fp32 lives only in registers and in the small per-split partials; nothing large is
materialized. No atomics, so results are deterministic.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _unpack_lanes(p, VPB: tl.constexpr, BITS: tl.constexpr):
        """[R, C] bytes -> [R, C, VPB] codes, lane-minor (lane i = bits i*BITS...)."""
        m: tl.constexpr = (1 << BITS) - 1
        if VPB == 2:
            return tl.join(p & m, (p >> 4) & m)
        else:  # 4 lanes of 2 bits
            # join(a, b) appends the new axis last: [.., j, o] = lane 2*j + o
            ev = tl.join(p & m, (p >> 4) & m)          # lanes 0, 2
            od = tl.join((p >> 2) & m, (p >> 6) & m)   # lanes 1, 3
            return tl.reshape(tl.join(ev, od), (p.shape[0], p.shape[1], 4))

    @triton.jit
    def _load_k(KQ, KS, KZ, KF, b, h, n0, offs_n, offs_d, NQ, nmask,
                skq_b, skq_h, skq_g, skq_d, skq_p, sks_b, sks_h, sks_g, sks_d,
                skf_b, skf_h, skf_n,
                KB: tl.constexpr, KG: tl.constexpr, HAS_Q: tl.constexpr, HAS_F: tl.constexpr,
                BN: tl.constexpr, D: tl.constexpr):
        """Key tile [BN, D] fp32 for tokens n0..n0+BN (n0 a multiple of KG)."""
        k = tl.zeros((BN, D), tl.float32)
        if HAS_Q:
            VPB: tl.constexpr = 8 // KB
            NG: tl.constexpr = BN // KG
            BY: tl.constexpr = KG // VPB
            r = tl.arange(0, NG * BY)                      # (group, byte) rows
            grp = n0 // KG + r // BY
            gm = grp * KG < NQ
            p = tl.load(KQ + b * skq_b + h * skq_h + grp[:, None] * skq_g + offs_d[None, :] * skq_d
                        + (r % BY)[:, None] * skq_p, mask=gm[:, None], other=0).to(tl.int32)
            code = _unpack_lanes(p, VPB, KB)               # [NG*BY, D, VPB]
            code = tl.reshape(tl.permute(code, (0, 2, 1)), (NG, KG, D))
            g2 = n0 // KG + tl.arange(0, NG)
            g2m = g2 * KG < NQ
            sc = tl.load(KS + b * sks_b + h * sks_h + g2[:, None] * sks_g + offs_d[None, :] * sks_d,
                         mask=g2m[:, None], other=0.0).to(tl.float32)
            zp = tl.load(KZ + b * sks_b + h * sks_h + g2[:, None] * sks_g + offs_d[None, :] * sks_d,
                         mask=g2m[:, None], other=0.0).to(tl.float32)
            kq = tl.reshape(code.to(tl.float32) * sc[:, None, :] + zp[:, None, :], (BN, D))
            k += tl.where((nmask & (offs_n < NQ))[:, None], kq, 0.0)
        if HAS_F:
            fm = nmask & (offs_n >= NQ)
            k += tl.load(KF + b * skf_b + h * skf_h + (offs_n - NQ)[:, None] * skf_n + offs_d[None, :],
                         mask=fm[:, None], other=0.0).to(tl.float32)
        return k

    @triton.jit
    def _load_v(VQ, VS, VZ, VF, b, h, n0, offs_n, offs_d, NQ, nmask,
                svq_b, svq_h, svq_n, svq_g, svq_p, svs_b, svs_h, svs_n, svs_g,
                svf_b, svf_h, svf_n,
                VB: tl.constexpr, VG: tl.constexpr, HAS_Q: tl.constexpr, HAS_F: tl.constexpr,
                BN: tl.constexpr, D: tl.constexpr):
        """Value tile [BN, D] fp32; values are packed per token over channel groups of VG."""
        v = tl.zeros((BN, D), tl.float32)
        if HAS_Q:
            VPB: tl.constexpr = 8 // VB
            NCG: tl.constexpr = D // VG
            BY: tl.constexpr = VG // VPB
            qm = nmask & (offs_n < NQ)
            c = tl.arange(0, NCG * BY)                     # (channel group, byte) columns
            p = tl.load(VQ + b * svq_b + h * svq_h + offs_n[:, None] * svq_n + (c // BY)[None, :] * svq_g
                        + (c % BY)[None, :] * svq_p, mask=qm[:, None], other=0).to(tl.int32)
            code = tl.reshape(_unpack_lanes(p, VPB, VB), (BN, NCG, VG))
            cg = tl.arange(0, NCG)
            sc = tl.load(VS + b * svs_b + h * svs_h + offs_n[:, None] * svs_n + cg[None, :] * svs_g,
                         mask=qm[:, None], other=0.0).to(tl.float32)
            zp = tl.load(VZ + b * svs_b + h * svs_h + offs_n[:, None] * svs_n + cg[None, :] * svs_g,
                         mask=qm[:, None], other=0.0).to(tl.float32)
            vq = tl.reshape(code.to(tl.float32) * sc[:, :, None] + zp[:, :, None], (BN, D))
            v += tl.where(qm[:, None], vq, 0.0)
        if HAS_F:
            fm = nmask & (offs_n >= NQ)
            v += tl.load(VF + b * svf_b + h * svf_h + (offs_n - NQ)[:, None] * svf_n + offs_d[None, :],
                         mask=fm[:, None], other=0.0).to(tl.float32)
        return v

    @triton.jit
    def _sk_fwd(Q, KQ, KS, KZ, VQ, VS, VZ, KF, VF, KC, VC, MO, LO, AO, MSK,
                sq_b, sq_h, sq_t,
                skq_b, skq_h, skq_g, skq_d, skq_p, sks_b, sks_h, sks_g, sks_d,
                svq_b, svq_h, svq_n, svq_g, svq_p, svs_b, svs_h, svs_n, svs_g,
                skf_b, skf_h, skf_n, svf_b, svf_h, svf_n,
                skc_b, skc_h, skc_n, svc_b, svc_h, svc_n,
                NQ, NF, NC, T, H_KV, NSPLIT, SPLIT_N, q_pos0, c_pos0, sm_scale,
                G: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                KB: tl.constexpr, VB: tl.constexpr, KG: tl.constexpr, VG: tl.constexpr,
                HAS_Q: tl.constexpr, HAS_F: tl.constexpr, CAUSAL: tl.constexpr, HAS_MASK: tl.constexpr):
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        sp = tl.program_id(2)
        b = pid_bh // H_KV
        h = pid_bh % H_KV
        R = G * T
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < R
        g = rows // T
        t = rows % T
        hq = h * G + g
        offs_d = tl.arange(0, D)
        q = tl.load(Q + b * sq_b + hq[:, None] * sq_h + t[:, None] * sq_t + offs_d[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.bfloat16)
        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        if sp < NSPLIT:
            start = sp * SPLIT_N
            end = tl.minimum(start + SPLIT_N, NQ + NF)
            for n0 in range(start, end, BLOCK_N):
                offs_n = n0 + tl.arange(0, BLOCK_N)
                nmask = offs_n < end
                k = _load_k(KQ, KS, KZ, KF, b, h, n0, offs_n, offs_d, NQ, nmask,
                            skq_b, skq_h, skq_g, skq_d, skq_p, sks_b, sks_h, sks_g, sks_d,
                            skf_b, skf_h, skf_n, KB, KG, HAS_Q, HAS_F, BLOCK_N, D)
                s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * sm_scale
                ok = nmask
                if HAS_MASK:
                    ok = ok & (tl.load(MSK + pid_bh * (NQ + NF) + offs_n, mask=nmask, other=0) != 0)
                s = tl.where(ok[None, :], s, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(s, 1))
                m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
                p = tl.exp(s - m_safe[:, None])
                alpha = tl.exp(m_i - m_safe)
                v = _load_v(VQ, VS, VZ, VF, b, h, n0, offs_n, offs_d, NQ, nmask,
                            svq_b, svq_h, svq_n, svq_g, svq_p, svs_b, svs_h, svs_n, svs_g,
                            svf_b, svf_h, svf_n, VB, VG, HAS_Q, HAS_F, BLOCK_N, D)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
                l_i = l_i * alpha + tl.sum(p, 1)
                m_i = m_new
        else:
            for n0 in range(0, NC, BLOCK_N):
                offs_n = n0 + tl.arange(0, BLOCK_N)
                nmask = offs_n < NC
                k = tl.load(KC + b * skc_b + h * skc_h + offs_n[:, None] * skc_n + offs_d[None, :],
                            mask=nmask[:, None], other=0.0)
                s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * sm_scale
                ok = nmask[None, :]
                if CAUSAL > 0:
                    ok = ok & (((c_pos0 + offs_n) // CAUSAL)[None, :] <= ((q_pos0 + t) // CAUSAL)[:, None])
                s = tl.where(ok, s, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(s, 1))
                m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
                p = tl.exp(s - m_safe[:, None])
                alpha = tl.exp(m_i - m_safe)
                v = tl.load(VC + b * svc_b + h * svc_h + offs_n[:, None] * svc_n + offs_d[None, :],
                            mask=nmask[:, None], other=0.0)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
                l_i = l_i * alpha + tl.sum(p, 1)
                m_i = m_new
        base = (pid_bh * (NSPLIT + 1) + sp) * R
        tl.store(MO + base + rows, m_i, mask=rmask)
        tl.store(LO + base + rows, l_i, mask=rmask)
        tl.store(AO + (base + rows)[:, None] * D + offs_d[None, :], acc, mask=rmask[:, None])

    @triton.jit
    def _sk_fused(Q, KQ, KS, KZ, VQ, VS, VZ, KF, VF, KC, VC, OUT, MP, LP, OP, MSK,
                  sq_b, sq_h, sq_t, so_b, so_h, so_t,
                  skq_b, skq_h, skq_g, skq_d, skq_p, sks_b, sks_h, sks_g, sks_d,
                  svq_b, svq_h, svq_n, svq_g, svq_p, svs_b, svs_h, svs_n, svs_g,
                  skf_b, skf_h, skf_n, svf_b, svf_h, svf_n,
                  skc_b, skc_h, skc_n, svc_b, svc_h, svc_n,
                  NQ, NF, NC, T, H_KV, q_pos0, c_pos0, sm_scale,
                  G: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  KB: tl.constexpr, VB: tl.constexpr, KG: tl.constexpr, VG: tl.constexpr,
                  HAS_Q: tl.constexpr, HAS_F: tl.constexpr, CAUSAL: tl.constexpr, WRITE_PREFIX: tl.constexpr,
                  HAS_MASK: tl.constexpr):
        """One program per (KV head, row tile): the whole prefix, then the current block."""
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        b = pid_bh // H_KV
        h = pid_bh % H_KV
        R = G * T
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < R
        g = rows // T
        t = rows % T
        hq = h * G + g
        offs_d = tl.arange(0, D)
        q = tl.load(Q + b * sq_b + hq[:, None] * sq_h + t[:, None] * sq_t + offs_d[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.bfloat16)
        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        for n0 in range(0, NQ + NF, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            nmask = offs_n < NQ + NF
            k = _load_k(KQ, KS, KZ, KF, b, h, n0, offs_n, offs_d, NQ, nmask,
                        skq_b, skq_h, skq_g, skq_d, skq_p, sks_b, sks_h, sks_g, sks_d,
                        skf_b, skf_h, skf_n, KB, KG, HAS_Q, HAS_F, BLOCK_N, D)
            s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * sm_scale
            ok = nmask
            if HAS_MASK:
                ok = ok & (tl.load(MSK + pid_bh * (NQ + NF) + offs_n, mask=nmask, other=0) != 0)
            s = tl.where(ok[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            v = _load_v(VQ, VS, VZ, VF, b, h, n0, offs_n, offs_d, NQ, nmask,
                        svq_b, svq_h, svq_n, svq_g, svq_p, svs_b, svs_h, svs_n, svs_g,
                        svf_b, svf_h, svf_n, VB, VG, HAS_Q, HAS_F, BLOCK_N, D)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new
        if WRITE_PREFIX:
            tl.store(MP + pid_bh * R + rows, m_i, mask=rmask)
            tl.store(LP + pid_bh * R + rows, l_i, mask=rmask)
            o_p = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
            tl.store(OP + (pid_bh * R + rows)[:, None] * D + offs_d[None, :], o_p.to(OP.dtype.element_ty),
                     mask=rmask[:, None])
        for n0 in range(0, NC, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            nmask = offs_n < NC
            k = tl.load(KC + b * skc_b + h * skc_h + offs_n[:, None] * skc_n + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * sm_scale
            ok = nmask[None, :]
            if CAUSAL > 0:
                ok = ok & (((c_pos0 + offs_n) // CAUSAL)[None, :] <= ((q_pos0 + t) // CAUSAL)[:, None])
            s = tl.where(ok, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            v = tl.load(VC + b * svc_b + h * svc_h + offs_n[:, None] * svc_n + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new
        out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        tl.store(OUT + b * so_b + hq[:, None] * so_h + t[:, None] * so_t + offs_d[None, :],
                 out.to(OUT.dtype.element_ty), mask=rmask[:, None])

    @triton.jit
    def _sk_combine(MO, LO, AO, OUT, MP, LP, OP, so_b, so_h, so_t,
                    T, H_KV, NSPLIT,
                    G: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, WRITE_PREFIX: tl.constexpr):
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        b = pid_bh // H_KV
        h = pid_bh % H_KV
        R = G * T
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < R
        offs_d = tl.arange(0, D)
        mp = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        lp = tl.zeros((BLOCK_M,), tl.float32)
        op = tl.zeros((BLOCK_M, D), tl.float32)
        for sp in range(0, NSPLIT):
            base = (pid_bh * (NSPLIT + 1) + sp) * R
            ms = tl.load(MO + base + rows, mask=rmask, other=float("-inf"))
            ls = tl.load(LO + base + rows, mask=rmask, other=0.0)
            a_s = tl.load(AO + (base + rows)[:, None] * D + offs_d[None, :], mask=rmask[:, None], other=0.0)
            mn = tl.maximum(mp, ms)
            m_safe = tl.where(mn == float("-inf"), 0.0, mn)
            ea = tl.exp(mp - m_safe)
            eb = tl.exp(ms - m_safe)
            lp = lp * ea + ls * eb
            op = op * ea[:, None] + a_s * eb[:, None]
            mp = mn
        base = (pid_bh * (NSPLIT + 1) + NSPLIT) * R
        mc = tl.load(MO + base + rows, mask=rmask, other=float("-inf"))
        lc = tl.load(LO + base + rows, mask=rmask, other=0.0)
        ac = tl.load(AO + (base + rows)[:, None] * D + offs_d[None, :], mask=rmask[:, None], other=0.0)
        m = tl.maximum(mp, mc)
        m_safe = tl.where(m == float("-inf"), 0.0, m)
        ea = tl.exp(mp - m_safe)
        eb = tl.exp(mc - m_safe)
        den = lp * ea + lc * eb
        out = (op * ea[:, None] + ac * eb[:, None]) / tl.where(den > 0, den, 1.0)[:, None]
        g = rows // T
        t = rows % T
        hq = h * G + g
        tl.store(OUT + b * so_b + hq[:, None] * so_h + t[:, None] * so_t + offs_d[None, :],
                 out.to(OUT.dtype.element_ty), mask=rmask[:, None])
        if WRITE_PREFIX:
            tl.store(MP + pid_bh * R + rows, mp, mask=rmask)
            tl.store(LP + pid_bh * R + rows, lp, mask=rmask)
            o_p = op / tl.where(lp > 0, lp, 1.0)[:, None]
            tl.store(OP + (pid_bh * R + rows)[:, None] * D + offs_d[None, :], o_p.to(OP.dtype.element_ty),
                     mask=rmask[:, None])

    @triton.jit
    def _sk_score(Q, QI, MP, LP, CEN, KQ, KS, KZ, VQ, VS, VZ, KF, VF, MASS, SCORE, MSK,
                  sq_b, sq_h, sq_t,
                  skq_b, skq_h, skq_g, skq_d, skq_p, sks_b, sks_h, sks_g, sks_d,
                  svq_b, svq_h, svq_n, svq_g, svq_p, svs_b, svs_h, svs_n, svs_g,
                  skf_b, skf_h, skf_n, svf_b, svf_h, svf_n,
                  NQ, NF, T, T2, H_KV, NOUT, sm_scale,
                  G: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  KB: tl.constexpr, VB: tl.constexpr, KG: tl.constexpr, VG: tl.constexpr,
                  HAS_Q: tl.constexpr, HAS_F: tl.constexpr, VALUE: tl.constexpr, HAS_MASK: tl.constexpr):
        pid_bh = tl.program_id(0)
        pid_n = tl.program_id(1)
        b = pid_bh // H_KV
        h = pid_bh % H_KV
        n0 = pid_n * BLOCK_N
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < NQ + NF
        offs_d = tl.arange(0, D)
        k = _load_k(KQ, KS, KZ, KF, b, h, n0, offs_n, offs_d, NQ, nmask,
                    skq_b, skq_h, skq_g, skq_d, skq_p, sks_b, sks_h, sks_g, sks_d,
                    skf_b, skf_h, skf_n, KB, KG, HAS_Q, HAS_F, BLOCK_N, D).to(tl.bfloat16)
        a = tl.zeros((BLOCK_N,), tl.float32)
        R2 = G * T2
        for r0 in range(0, R2, BLOCK_M):
            rows = r0 + tl.arange(0, BLOCK_M)
            rm = rows < R2
            g = rows // T2
            t = tl.load(QI + rows % T2, mask=rm, other=0)
            hq = h * G + g
            q = tl.load(Q + b * sq_b + hq[:, None] * sq_h + t[:, None] * sq_t + offs_d[None, :],
                        mask=rm[:, None], other=0.0).to(tl.bfloat16)
            rf = g * T + t
            mp = tl.load(MP + pid_bh * G * T + rf, mask=rm, other=0.0)
            lp = tl.load(LP + pid_bh * G * T + rf, mask=rm, other=1.0)
            s = tl.dot(q, tl.trans(k)) * sm_scale
            p = tl.exp(s - mp[:, None]) / tl.where(lp > 0, lp, 1.0)[:, None]
            p = tl.where(rm[:, None] & nmask[None, :], p, 0.0)
            a += tl.sum(p, 0)
        a = a / R2
        if HAS_MASK:
            a = tl.where(tl.load(MSK + pid_bh * NOUT + offs_n, mask=nmask, other=0) != 0, a, 0.0)
        out_n = pid_bh * NOUT + offs_n
        tl.store(MASS + out_n, a, mask=nmask)
        if VALUE:
            v = _load_v(VQ, VS, VZ, VF, b, h, n0, offs_n, offs_d, NQ, nmask,
                        svq_b, svq_h, svq_n, svq_g, svq_p, svs_b, svs_h, svs_n, svs_g,
                        svf_b, svf_h, svf_n, VB, VG, HAS_Q, HAS_F, BLOCK_N, D)
            c = tl.load(CEN + pid_bh * D + offs_d).to(tl.float32)
            dv = v - c[None, :]
            dist = tl.sqrt(tl.sum(dv * dv, 1))
            tl.store(SCORE + out_n, a * dist, mask=nmask)


    @triton.jit
    def _sk_dequant(KQ, KS, KZ, VQ, VS, VZ, KF, VF, KO, VO, KCH, sch_b, sch_h,
                    skq_b, skq_h, skq_g, skq_d, skq_p, sks_b, sks_h, sks_g, sks_d,
                    svq_b, svq_h, svq_n, svq_g, svq_p, svs_b, svs_h, svs_n, svs_g,
                    skf_b, skf_h, skf_n, svf_b, svf_h, svf_n,
                    NQ, NF, H_KV,
                    D: tl.constexpr, BLOCK_N: tl.constexpr, KB: tl.constexpr, VB: tl.constexpr,
                    KG: tl.constexpr, VG: tl.constexpr, HAS_Q: tl.constexpr, HAS_F: tl.constexpr,
                    KMODE: tl.constexpr):
        pid_bh = tl.program_id(0)
        n0 = tl.program_id(1) * BLOCK_N
        b = pid_bh // H_KV
        h = pid_bh % H_KV
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < NQ + NF
        offs_d = tl.arange(0, D)
        if KMODE == 1:
            # per-token keys (laid out like values), times the per-channel divisor c
            k = _load_v(KQ, KS, KZ, KF, b, h, n0, offs_n, offs_d, NQ, nmask,
                        skq_b, skq_h, skq_g, skq_d, skq_p, sks_b, sks_h, sks_g, sks_d,
                        skf_b, skf_h, skf_n, KB, VG, HAS_Q, HAS_F, BLOCK_N, D)
            cch = tl.load(KCH + b * sch_b + h * sch_h + offs_d)
            k = tl.where((offs_n < NQ)[:, None], k * cch[None, :], k)      # residual stays raw bf16
        else:
            k = _load_k(KQ, KS, KZ, KF, b, h, n0, offs_n, offs_d, NQ, nmask,
                        skq_b, skq_h, skq_g, skq_d, skq_p, sks_b, sks_h, sks_g, sks_d,
                        skf_b, skf_h, skf_n, KB, KG, HAS_Q, HAS_F, BLOCK_N, D)
        v = _load_v(VQ, VS, VZ, VF, b, h, n0, offs_n, offs_d, NQ, nmask,
                    svq_b, svq_h, svq_n, svq_g, svq_p, svs_b, svs_h, svs_n, svs_g,
                    svf_b, svf_h, svf_n, VB, VG, HAS_Q, HAS_F, BLOCK_N, D)
        o = (pid_bh * (NQ + NF) + offs_n)[:, None] * D + offs_d[None, :]
        tl.store(KO + o, k.to(KO.dtype.element_ty), mask=nmask[:, None])
        tl.store(VO + o, v.to(VO.dtype.element_ty), mask=nmask[:, None])


class Prefix:
    """The prefix a kernel reads: packed (+ bf16 residual tail) or plain bf16. [B, Hkv, n, D]."""

    __slots__ = ("kq", "ks", "kz", "vq", "vs", "vz", "kf", "vf", "nq", "nf", "kb", "vb", "kg", "vg", "mask",
                 "kmode", "kch")

    def __init__(self, *, kq=None, ks=None, kz=None, vq=None, vs=None, vz=None, kf=None, vf=None,
                 nq=0, nf=0, kb=16, vb=16, kg=32, vg=32, mask=None, kmode=0, kch=None):
        self.mask = mask          # optional uint8 [B, Hkv, n] (contiguous): 0 = dead slot, never attended
        self.kmode, self.kch = kmode, kch    # kmode 1: per-token keys with divisor kch [B, Hkv, D] fp32
        self.kq, self.ks, self.kz, self.vq, self.vs, self.vz = kq, ks, kz, vq, vs, vz
        self.kf, self.vf, self.nq, self.nf = kf, vf, int(nq), int(nf)
        self.kb, self.vb, self.kg, self.vg = kb, vb, kg, vg

    @property
    def n(self) -> int:
        return self.nq + self.nf

    @classmethod
    def from_view(cls, view) -> "Prefix":
        if view.k_bits == 16 and view.v_bits == 16:
            return cls(kf=view.k_fp, vf=view.v_fp, nf=view.length)
        if view.k_bits == 16 or view.v_bits == 16:
            raise NotImplementedError("mixed 16-bit / packed K,V")
        return cls(kq=view.k_q, ks=view.k_scale, kz=view.k_zero, vq=view.v_q, vs=view.v_scale, vz=view.v_zero,
                   kf=view.k_residual, vf=view.v_residual, nq=view.quantized_length, nf=view.residual_length,
                   kb=view.k_bits, vb=view.v_bits, kg=view.key_token_group, vg=view.value_channel_group,
                   kmode=1 if getattr(view, "key_mode", "channel") == "token" else 0,
                   kch=getattr(view, "k_chan", None))

    @classmethod
    def dense(cls, key: torch.Tensor, value: torch.Tensor, mask=None) -> "Prefix":
        return cls(kf=key, vf=value, nf=key.shape[2], mask=mask)

    def args(self, dummy: torch.Tensor):
        """Pointer and stride arguments shared by all kernels (dummies for absent parts)."""
        z6 = (0,) * 5
        if self.nq:
            kq, ks, kz, vq, vs, vz = self.kq, self.ks, self.kz, self.vq, self.vs, self.vz
            skq = tuple(kq.stride()[:5])
            sks = tuple(ks.stride()[:4])
            svq = tuple(vq.stride()[:5])
            svs = tuple(vs.stride()[:4])
        else:
            kq = ks = kz = vq = vs = vz = dummy
            skq, sks, svq, svs = z6, (0,) * 4, z6, (0,) * 4
        if self.nf:
            kf, vf = self.kf, self.vf
            assert kf.stride(-1) == 1 and vf.stride(-1) == 1
            skf = tuple(kf.stride()[:3])
            svf = tuple(vf.stride()[:3])
        else:
            kf = vf = dummy
            skf = svf = (0, 0, 0)
        return (kq, ks, kz, vq, vs, vz, kf, vf), skq, sks, svq, svs, skf, svf


def dequantize_prefix(prefix: Prefix, b: int, hkv: int, d: int, dtype=torch.bfloat16, block_n: int = 64):
    """The whole prefix as contiguous [B, Hkv, n, D] tensors (one kernel)."""
    n = prefix.n
    ko = torch.empty((b, hkv, n, d), device=(prefix.kq if prefix.nq else prefix.kf).device, dtype=dtype)
    vo = torch.empty_like(ko)
    if n == 0:
        return ko, vo
    ptrs, skq, sks, svq, svs, skf, svf = prefix.args(ko)
    kch = prefix.kch if (prefix.kmode and prefix.kch is not None) else ko
    sch = (kch.stride(0), kch.stride(1)) if prefix.kmode else (0, 0)
    _sk_dequant[(b * hkv, triton.cdiv(n, block_n))](
        *ptrs, ko, vo, kch, *sch, *skq, *sks, *svq, *svs, *skf, *svf, prefix.nq, prefix.nf, hkv,
        D=d, BLOCK_N=block_n, KB=prefix.kb, VB=prefix.vb, KG=prefix.kg, VG=prefix.vg,
        HAS_Q=prefix.nq > 0, HAS_F=prefix.nf > 0, KMODE=prefix.kmode, num_warps=4,
    )
    return ko, vo


def _num_sms(device) -> int:
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 80


_SMS: dict = {}


def splitk_attention(query, prefix: Prefix, current_key, current_value, *, scaling=None,
                     causal_block: int = 0, q_pos0: int = 0, c_pos0: int = 0, want_prefix: bool = False,
                     block_m: int = 64, block_n: int = 32, num_warps: int = 4,
                     dequant_rows: int = 256, packed_direct_max: int = 256, fused_max: int = 1200):
    """softmax over [prefix | current] for every query row. Returns (out, stats).

    stats (want_prefix) = (M_p, L_p, O_p, prefix): prefix-only max, sum and normalized
    output per row r = g*T + t of each KV head ([B, Hkv, R], [B, Hkv, R], [B, Hkv, R, D]),
    and the prefix actually read (unpacked to bf16 when there were many rows).
    causal_block > 0: a query at position q_pos0+t sees current key c_pos0+j only if
    (c_pos0+j)//causal_block <= (q_pos0+t)//causal_block.

    Dispatch (measured on PPU): a packed prefix longer than packed_direct_max, or read
    by more than dequant_rows rows, is unpacked to bf16 first; a prefix of at most
    fused_max entries runs in one fused launch, a longer one in split-K + combine.
    """
    b, hq, t, d = query.shape
    hkv = current_key.shape[1]
    g = hq // hkv
    r = g * t
    if prefix.nq and (r > dequant_rows or prefix.n > packed_direct_max or prefix.kmode):
        # Many query rows (every row tile would unpack the prefix again) or a long
        # prefix (the bf16 path is faster there): unpack once, one cheap kernel.
        prefix = Prefix.dense(*dequantize_prefix(prefix, b, hkv, d, dtype=query.dtype), mask=prefix.mask)
    assert block_n % prefix.kg == 0
    msk = prefix.mask if prefix.mask is not None else query
    assert query.stride(-1) == 1 and current_key.stride(-1) == 1 and current_value.stride(-1) == 1
    dev = query.device
    sms = _SMS.setdefault(dev, _num_sms(dev))
    n = prefix.n
    row_tiles = triton.cdiv(r, block_m)
    progs = b * hkv * row_tiles
    target = max(1, (2 * sms) // max(progs, 1))
    nsplit = min(target, max(1, triton.cdiv(n, block_n))) if n else 0
    split_n = triton.cdiv(triton.cdiv(n, nsplit), block_n) * block_n if nsplit else block_n
    nsplit = triton.cdiv(n, split_n) if n else 0
    scale = scaling if scaling is not None else d ** -0.5
    if n <= fused_max or nsplit <= 1:
        # Short prefix: one launch, no partials (a launch costs ~60 us of host time here).
        out = torch.empty_like(query)
        if want_prefix:
            mp = torch.empty((b, hkv, r), device=dev, dtype=torch.float32)
            lp = torch.empty_like(mp)
            op = torch.empty((b, hkv, r, d), device=dev, dtype=query.dtype)
        else:
            mp = lp = op = out
        ptrs, skq, sks, svq, svs, skf, svf = prefix.args(out)
        _sk_fused[(b * hkv, row_tiles)](
            query, *ptrs, current_key, current_value, out, mp, lp, op, msk,
            query.stride(0), query.stride(1), query.stride(2), out.stride(0), out.stride(1), out.stride(2),
            *skq, *sks, *svq, *svs, *skf, *svf,
            current_key.stride(0), current_key.stride(1), current_key.stride(2),
            current_value.stride(0), current_value.stride(1), current_value.stride(2),
            prefix.nq, prefix.nf, current_key.shape[2], t, hkv, q_pos0, c_pos0, scale,
            G=g, D=d, BLOCK_M=block_m, BLOCK_N=block_n, KB=prefix.kb, VB=prefix.vb, KG=prefix.kg,
            VG=prefix.vg, HAS_Q=prefix.nq > 0, HAS_F=prefix.nf > 0, CAUSAL=causal_block,
            WRITE_PREFIX=want_prefix, HAS_MASK=prefix.mask is not None, num_warps=num_warps,
        )
        return out, ((mp, lp, op, prefix) if want_prefix else None)
    mo = torch.empty((b * hkv * (nsplit + 1) * r,), device=dev, dtype=torch.float32)
    lo = torch.empty_like(mo)
    ao = torch.empty((b * hkv * (nsplit + 1) * r, d), device=dev, dtype=torch.float32)
    ptrs, skq, sks, svq, svs, skf, svf = prefix.args(mo)
    scale = scaling if scaling is not None else d ** -0.5
    _sk_fwd[(b * hkv, row_tiles, nsplit + 1)](
        query, *ptrs, current_key, current_value, mo, lo, ao, msk,
        query.stride(0), query.stride(1), query.stride(2),
        *skq, *sks, *svq, *svs, *skf, *svf,
        current_key.stride(0), current_key.stride(1), current_key.stride(2),
        current_value.stride(0), current_value.stride(1), current_value.stride(2),
        prefix.nq, prefix.nf, current_key.shape[2], t, hkv, nsplit, split_n, q_pos0, c_pos0, scale,
        G=g, D=d, BLOCK_M=block_m, BLOCK_N=block_n, KB=prefix.kb, VB=prefix.vb, KG=prefix.kg, VG=prefix.vg,
        HAS_Q=prefix.nq > 0, HAS_F=prefix.nf > 0, CAUSAL=causal_block, HAS_MASK=prefix.mask is not None,
        num_warps=num_warps,
    )
    out = torch.empty_like(query)
    if want_prefix:
        mp = torch.empty((b, hkv, r), device=dev, dtype=torch.float32)
        lp = torch.empty_like(mp)
        op = torch.empty((b, hkv, r, d), device=dev, dtype=query.dtype)
    else:
        mp = lp = op = mo
    _sk_combine[(b * hkv, row_tiles)](
        mo, lo, ao, out, mp, lp, op, out.stride(0), out.stride(1), out.stride(2),
        t, hkv, nsplit, G=g, D=d, BLOCK_M=block_m, WRITE_PREFIX=want_prefix, num_warps=4,
    )
    return out, ((mp, lp, op, prefix) if want_prefix else None)


def prefix_scores(query, prefix: Prefix, stats, q_rows: torch.Tensor, *, value: bool, scaling=None,
                  block_m: int = 64, block_n: int = 64):
    """Per prefix entry: mass a_j (mean prefix-softmax over the selection rows) and, with
    value=True, a_j * ||v_j - c|| with c = sum_j a_j v_j (= mean of O_p over the rows).

    q_rows: int32 positions t (within the block) of the selection queries; every query
    head of the KV head contributes. Returns (mass, score) as [B, Hkv, n] fp32 (score
    None when value=False)."""
    b, hq, t, d = query.shape
    mp, lp, op, used = stats
    prefix = used if used is not None else prefix          # reuse the unpacked prefix
    hkv = mp.shape[1]
    g = hq // hkv
    n = prefix.n
    t2 = int(q_rows.numel())
    mass = torch.empty((b, hkv, n), device=query.device, dtype=torch.float32)
    score = torch.empty_like(mass) if value else mass
    if value:
        rows = (torch.arange(g, device=query.device).unsqueeze(1) * t + q_rows.to(torch.int64).unsqueeze(0)).reshape(-1)
        cen = op.index_select(2, rows).float().mean(2)          # [B, Hkv, D], tiny
    else:
        cen = mass
    ptrs, skq, sks, svq, svs, skf, svf = prefix.args(mass)
    scale = scaling if scaling is not None else d ** -0.5
    _sk_score[(b * hkv, triton.cdiv(n, block_n))](
        query, q_rows, mp, lp, cen, *ptrs, mass, score, prefix.mask if prefix.mask is not None else mass,
        query.stride(0), query.stride(1), query.stride(2),
        *skq, *sks, *svq, *svs, *skf, *svf,
        prefix.nq, prefix.nf, t, t2, hkv, n, scale,
        G=g, D=d, BLOCK_M=block_m, BLOCK_N=block_n, KB=prefix.kb, VB=prefix.vb, KG=prefix.kg, VG=prefix.vg,
        HAS_Q=prefix.nq > 0, HAS_F=prefix.nf > 0, VALUE=value, HAS_MASK=prefix.mask is not None, num_warps=4,
    )
    return mass, (score if value else None)
