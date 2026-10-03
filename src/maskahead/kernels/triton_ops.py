from __future__ import annotations

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _load_packed_key_tile(
        kq_ptr,
        ks_ptr,
        kz_ptr,
        b,
        h,
        group_id,
        token_in_group,
        offs_d,
        stride_kqb,
        stride_kqh,
        stride_kqg,
        stride_kqd,
        stride_kqp,
        stride_ksb,
        stride_ksh,
        stride_ksg,
        stride_ksd,
        KEY_BITS: tl.constexpr,
        KEY_GROUP: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        values_per_byte: tl.constexpr = 8 // KEY_BITS
        byte_id = token_in_group // values_per_byte
        lane = token_in_group % values_per_byte
        p = tl.load(
            kq_ptr
            + b * stride_kqb
            + h * stride_kqh
            + group_id * stride_kqg
            + offs_d * stride_kqd
            + byte_id * stride_kqp,
            mask=offs_d < HEAD_DIM,
            other=0,
        ).to(tl.int32)
        q = (p >> (lane * KEY_BITS)) & ((1 << KEY_BITS) - 1)
        scale = tl.load(
            ks_ptr
            + b * stride_ksb
            + h * stride_ksh
            + group_id * stride_ksg
            + offs_d * stride_ksd,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            kz_ptr
            + b * stride_ksb
            + h * stride_ksh
            + group_id * stride_ksg
            + offs_d * stride_ksd,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        return q.to(tl.float32) * scale + zero


    @triton.jit
    def _load_packed_value_tile(
        vq_ptr,
        vs_ptr,
        vz_ptr,
        b,
        h,
        token_id,
        offs_d,
        stride_vqb,
        stride_vqh,
        stride_vqn,
        stride_vqg,
        stride_vqp,
        stride_vsb,
        stride_vsh,
        stride_vsn,
        stride_vsg,
        VALUE_BITS: tl.constexpr,
        VALUE_GROUP: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        values_per_byte: tl.constexpr = 8 // VALUE_BITS
        channel_group = offs_d // VALUE_GROUP
        channel_in_group = offs_d % VALUE_GROUP
        byte_id = channel_in_group // values_per_byte
        lane = channel_in_group % values_per_byte
        p = tl.load(
            vq_ptr
            + b * stride_vqb
            + h * stride_vqh
            + token_id * stride_vqn
            + channel_group * stride_vqg
            + byte_id * stride_vqp,
            mask=offs_d < HEAD_DIM,
            other=0,
        ).to(tl.int32)
        q = (p >> (lane * VALUE_BITS)) & ((1 << VALUE_BITS) - 1)
        scale = tl.load(
            vs_ptr
            + b * stride_vsb
            + h * stride_vsh
            + token_id * stride_vsn
            + channel_group * stride_vsg,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            vz_ptr
            + b * stride_vsb
            + h * stride_vsh
            + token_id * stride_vsn
            + channel_group * stride_vsg,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        return q.to(tl.float32) * scale + zero


    @triton.jit
    def dense_packed_attention_kernel(
        q_ptr,
        kq_ptr,
        ks_ptr,
        kz_ptr,
        vq_ptr,
        vs_ptr,
        vz_ptr,
        kfp_ptr,
        vfp_ptr,
        kr_ptr,
        vr_ptr,
        kc_ptr,
        vc_ptr,
        out_ptr,
        n_quant,
        n_residual,
        n_current,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kqb,
        stride_kqh,
        stride_kqg,
        stride_kqd,
        stride_kqp,
        stride_ksb,
        stride_ksh,
        stride_ksg,
        stride_ksd,
        stride_vqb,
        stride_vqh,
        stride_vqn,
        stride_vqg,
        stride_vqp,
        stride_vsb,
        stride_vsh,
        stride_vsn,
        stride_vsg,
        stride_kfb,
        stride_kfh,
        stride_kfn,
        stride_kfd,
        stride_vfb,
        stride_vfh,
        stride_vfn,
        stride_vfd,
        stride_krb,
        stride_krh,
        stride_krn,
        stride_krd,
        stride_vrb,
        stride_vrh,
        stride_vrn,
        stride_vrd,
        stride_kcb,
        stride_kch,
        stride_kcn,
        stride_kcd,
        stride_vcb,
        stride_vch,
        stride_vcn,
        stride_vcd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        sm_scale,
        H_Q: tl.constexpr,
        H_KV: tl.constexpr,
        T_Q: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        KEY_BITS: tl.constexpr,
        VALUE_BITS: tl.constexpr,
        KEY_GROUP: tl.constexpr,
        VALUE_GROUP: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        b = pid_bh // H_Q
        hq = pid_bh % H_Q
        hkv = hq // (H_Q // H_KV)
        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(
            q_ptr
            + b * stride_qb
            + hq * stride_qh
            + pid_m * stride_qm
            + offs_d * stride_qd,
            mask=(pid_m < T_Q) & (offs_d < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), tl.float32)


        for start_n in tl.range(0, n_quant, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < n_quant
            if KEY_BITS == 16:
                k = tl.load(
                    kfp_ptr
                    + b * stride_kfb
                    + hkv * stride_kfh
                    + offs_n[:, None] * stride_kfn
                    + offs_d[None, :] * stride_kfd,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
            else:
                key_group = offs_n // KEY_GROUP
                token_in_group = offs_n % KEY_GROUP
                values_per_byte: tl.constexpr = 8 // KEY_BITS
                byte_id = token_in_group // values_per_byte
                lane = token_in_group % values_per_byte
                p = tl.load(
                    kq_ptr
                    + b * stride_kqb
                    + hkv * stride_kqh
                    + key_group[:, None] * stride_kqg
                    + offs_d[None, :] * stride_kqd
                    + byte_id[:, None] * stride_kqp,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0,
                ).to(tl.int32)
                qi = (p >> (lane[:, None] * KEY_BITS)) & ((1 << KEY_BITS) - 1)
                sc = tl.load(
                    ks_ptr
                    + b * stride_ksb
                    + hkv * stride_ksh
                    + key_group[:, None] * stride_ksg
                    + offs_d[None, :] * stride_ksd,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
                ze = tl.load(
                    kz_ptr
                    + b * stride_ksb
                    + hkv * stride_ksh
                    + key_group[:, None] * stride_ksg
                    + offs_d[None, :] * stride_ksd,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
                k = qi.to(tl.float32) * sc + ze

            scores = tl.sum(k * q[None, :], axis=1) * sm_scale
            scores = tl.where(valid_n, scores, -float("inf"))

            if VALUE_BITS == 16:
                v = tl.load(
                    vfp_ptr
                    + b * stride_vfb
                    + hkv * stride_vfh
                    + offs_n[:, None] * stride_vfn
                    + offs_d[None, :] * stride_vfd,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
            else:
                values_per_byte_v: tl.constexpr = 8 // VALUE_BITS
                channel_group = offs_d // VALUE_GROUP
                channel_in_group = offs_d % VALUE_GROUP
                byte_id_v = channel_in_group // values_per_byte_v
                lane_v = channel_in_group % values_per_byte_v
                pv = tl.load(
                    vq_ptr
                    + b * stride_vqb
                    + hkv * stride_vqh
                    + offs_n[:, None] * stride_vqn
                    + channel_group[None, :] * stride_vqg
                    + byte_id_v[None, :] * stride_vqp,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0,
                ).to(tl.int32)
                qv = (pv >> (lane_v[None, :] * VALUE_BITS)) & ((1 << VALUE_BITS) - 1)
                scv = tl.load(
                    vs_ptr
                    + b * stride_vsb
                    + hkv * stride_vsh
                    + offs_n[:, None] * stride_vsn
                    + channel_group[None, :] * stride_vsg,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
                zev = tl.load(
                    vz_ptr
                    + b * stride_vsb
                    + hkv * stride_vsh
                    + offs_n[:, None] * stride_vsn
                    + channel_group[None, :] * stride_vsg,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
                v = qv.to(tl.float32) * scv + zev

            m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_ij)
            p_attn = tl.exp(scores - m_ij)
            l_i = l_i * alpha + tl.sum(p_attn, axis=0)
            acc = acc * alpha + tl.sum(p_attn[:, None] * v, axis=0)
            m_i = m_ij


        for start_n in tl.range(0, n_residual, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < n_residual
            if KEY_BITS == 16:
                k = tl.load(
                    kfp_ptr
                    + b * stride_kfb
                    + hkv * stride_kfh
                    + (n_quant + offs_n)[:, None] * stride_kfn
                    + offs_d[None, :] * stride_kfd,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
            else:
                k = tl.load(
                    kr_ptr
                    + b * stride_krb
                    + hkv * stride_krh
                    + offs_n[:, None] * stride_krn
                    + offs_d[None, :] * stride_krd,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
            if VALUE_BITS == 16:
                v = tl.load(
                    vfp_ptr
                    + b * stride_vfb
                    + hkv * stride_vfh
                    + (n_quant + offs_n)[:, None] * stride_vfn
                    + offs_d[None, :] * stride_vfd,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
            else:
                v = tl.load(
                    vr_ptr
                    + b * stride_vrb
                    + hkv * stride_vrh
                    + offs_n[:, None] * stride_vrn
                    + offs_d[None, :] * stride_vrd,
                    mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
            scores = tl.sum(k * q[None, :], axis=1) * sm_scale
            scores = tl.where(valid_n, scores, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_ij)
            p_attn = tl.exp(scores - m_ij)
            l_i = l_i * alpha + tl.sum(p_attn, axis=0)
            acc = acc * alpha + tl.sum(p_attn[:, None] * v, axis=0)
            m_i = m_ij


        for start_n in tl.range(0, n_current, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < n_current
            k = tl.load(
                kc_ptr
                + b * stride_kcb
                + hkv * stride_kch
                + offs_n[:, None] * stride_kcn
                + offs_d[None, :] * stride_kcd,
                mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                vc_ptr
                + b * stride_vcb
                + hkv * stride_vch
                + offs_n[:, None] * stride_vcn
                + offs_d[None, :] * stride_vcd,
                mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(k * q[None, :], axis=1) * sm_scale
            scores = tl.where(valid_n, scores, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_ij)
            p_attn = tl.exp(scores - m_ij)
            l_i = l_i * alpha + tl.sum(p_attn, axis=0)
            acc = acc * alpha + tl.sum(p_attn[:, None] * v, axis=0)
            m_i = m_ij

        out = acc / l_i
        tl.store(
            out_ptr
            + b * stride_ob
            + hq * stride_oh
            + pid_m * stride_om
            + offs_d * stride_od,
            out,
            mask=(pid_m < T_Q) & (offs_d < HEAD_DIM),
        )

    @triton.jit
    def dense_packed_attention_blocked_kernel(
        q_ptr,
        kq_ptr,
        ks_ptr,
        kz_ptr,
        vq_ptr,
        vs_ptr,
        vz_ptr,
        kfp_ptr,
        vfp_ptr,
        kr_ptr,
        vr_ptr,
        kc_ptr,
        vc_ptr,
        out_ptr,
        n_quant,
        n_residual,
        n_current,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kqb,
        stride_kqh,
        stride_kqg,
        stride_kqd,
        stride_kqp,
        stride_ksb,
        stride_ksh,
        stride_ksg,
        stride_ksd,
        stride_vqb,
        stride_vqh,
        stride_vqn,
        stride_vqg,
        stride_vqp,
        stride_vsb,
        stride_vsh,
        stride_vsn,
        stride_vsg,
        stride_kfb,
        stride_kfh,
        stride_kfn,
        stride_kfd,
        stride_vfb,
        stride_vfh,
        stride_vfn,
        stride_vfd,
        stride_krb,
        stride_krh,
        stride_krn,
        stride_krd,
        stride_vrb,
        stride_vrh,
        stride_vrn,
        stride_vrd,
        stride_kcb,
        stride_kch,
        stride_kcn,
        stride_kcd,
        stride_vcb,
        stride_vch,
        stride_vcn,
        stride_vcd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        sm_scale,
        H_Q: tl.constexpr,
        H_KV: tl.constexpr,
        T_Q: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        KEY_BITS: tl.constexpr,
        VALUE_BITS: tl.constexpr,
        KEY_GROUP: tl.constexpr,
        VALUE_GROUP: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_mt = tl.program_id(1)
        b = pid_bh // H_Q
        hq = pid_bh % H_Q
        hkv = hq // (H_Q // H_KV)
        offs_m = pid_mt * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        valid_m = offs_m < T_Q
        valid_d = offs_d < HEAD_DIM
        q = tl.load(
            q_ptr
            + b * stride_qb
            + hq * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd,
            mask=valid_m[:, None] & valid_d[None, :],
            other=0.0,
        )
        if USE_BF16:
            q_dot = q.to(tl.bfloat16)
        else:
            q_dot = q.to(tl.float16)

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)


        for start_n in tl.range(0, n_quant, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < n_quant
            if KEY_BITS == 16:
                k = tl.load(
                    kfp_ptr
                    + b * stride_kfb
                    + hkv * stride_kfh
                    + offs_n[:, None] * stride_kfn
                    + offs_d[None, :] * stride_kfd,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0.0,
                )
            else:
                key_group = offs_n // KEY_GROUP
                token_in_group = offs_n % KEY_GROUP
                values_per_byte: tl.constexpr = 8 // KEY_BITS
                byte_id = token_in_group // values_per_byte
                lane = token_in_group % values_per_byte
                packed = tl.load(
                    kq_ptr
                    + b * stride_kqb
                    + hkv * stride_kqh
                    + key_group[:, None] * stride_kqg
                    + offs_d[None, :] * stride_kqd
                    + byte_id[:, None] * stride_kqp,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0,
                ).to(tl.int32)
                qk_i = (packed >> (lane[:, None] * KEY_BITS)) & ((1 << KEY_BITS) - 1)
                scale_k = tl.load(
                    ks_ptr
                    + b * stride_ksb
                    + hkv * stride_ksh
                    + key_group[:, None] * stride_ksg
                    + offs_d[None, :] * stride_ksd,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                zero_k = tl.load(
                    kz_ptr
                    + b * stride_ksb
                    + hkv * stride_ksh
                    + key_group[:, None] * stride_ksg
                    + offs_d[None, :] * stride_ksd,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                k = qk_i.to(tl.float32) * scale_k + zero_k
            if USE_BF16:
                k_dot = k.to(tl.bfloat16)
            else:
                k_dot = k.to(tl.float16)
            scores = tl.dot(q_dot, tl.trans(k_dot), out_dtype=tl.float32) * sm_scale
            scores = tl.where(valid_n[None, :], scores, -float("inf"))
            scores = tl.where(valid_m[:, None], scores, 0.0)

            if VALUE_BITS == 16:
                v = tl.load(
                    vfp_ptr
                    + b * stride_vfb
                    + hkv * stride_vfh
                    + offs_n[:, None] * stride_vfn
                    + offs_d[None, :] * stride_vfd,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0.0,
                )
            else:
                values_per_byte_v: tl.constexpr = 8 // VALUE_BITS
                channel_group = offs_d // VALUE_GROUP
                channel_in_group = offs_d % VALUE_GROUP
                byte_id_v = channel_in_group // values_per_byte_v
                lane_v = channel_in_group % values_per_byte_v
                packed_v = tl.load(
                    vq_ptr
                    + b * stride_vqb
                    + hkv * stride_vqh
                    + offs_n[:, None] * stride_vqn
                    + channel_group[None, :] * stride_vqg
                    + byte_id_v[None, :] * stride_vqp,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0,
                ).to(tl.int32)
                qv_i = (packed_v >> (lane_v[None, :] * VALUE_BITS)) & ((1 << VALUE_BITS) - 1)
                scale_v = tl.load(
                    vs_ptr
                    + b * stride_vsb
                    + hkv * stride_vsh
                    + offs_n[:, None] * stride_vsn
                    + channel_group[None, :] * stride_vsg,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                zero_v = tl.load(
                    vz_ptr
                    + b * stride_vsb
                    + hkv * stride_vsh
                    + offs_n[:, None] * stride_vsn
                    + channel_group[None, :] * stride_vsg,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                v = qv_i.to(tl.float32) * scale_v + zero_v

            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(scores - m_ij[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            if USE_BF16:
                pv = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), out_dtype=tl.float32)
            else:
                pv = tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32)
            acc = acc * alpha[:, None] + pv
            m_i = m_ij


        for start_n in tl.range(0, n_residual, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < n_residual
            if KEY_BITS == 16:
                k = tl.load(
                    kfp_ptr
                    + b * stride_kfb
                    + hkv * stride_kfh
                    + (n_quant + offs_n)[:, None] * stride_kfn
                    + offs_d[None, :] * stride_kfd,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0.0,
                )
            else:
                k = tl.load(
                    kr_ptr
                    + b * stride_krb
                    + hkv * stride_krh
                    + offs_n[:, None] * stride_krn
                    + offs_d[None, :] * stride_krd,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0.0,
                )
            if VALUE_BITS == 16:
                v = tl.load(
                    vfp_ptr
                    + b * stride_vfb
                    + hkv * stride_vfh
                    + (n_quant + offs_n)[:, None] * stride_vfn
                    + offs_d[None, :] * stride_vfd,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0.0,
                )
            else:
                v = tl.load(
                    vr_ptr
                    + b * stride_vrb
                    + hkv * stride_vrh
                    + offs_n[:, None] * stride_vrn
                    + offs_d[None, :] * stride_vrd,
                    mask=valid_n[:, None] & valid_d[None, :],
                    other=0.0,
                )
            if USE_BF16:
                k_dot = k.to(tl.bfloat16)
            else:
                k_dot = k.to(tl.float16)
            scores = tl.dot(q_dot, tl.trans(k_dot), out_dtype=tl.float32) * sm_scale
            scores = tl.where(valid_n[None, :], scores, -float("inf"))
            scores = tl.where(valid_m[:, None], scores, 0.0)
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(scores - m_ij[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            if USE_BF16:
                pv = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), out_dtype=tl.float32)
            else:
                pv = tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32)
            acc = acc * alpha[:, None] + pv
            m_i = m_ij


        for start_n in tl.range(0, n_current, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < n_current
            k = tl.load(
                kc_ptr
                + b * stride_kcb
                + hkv * stride_kch
                + offs_n[:, None] * stride_kcn
                + offs_d[None, :] * stride_kcd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            )
            v = tl.load(
                vc_ptr
                + b * stride_vcb
                + hkv * stride_vch
                + offs_n[:, None] * stride_vcn
                + offs_d[None, :] * stride_vcd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            )
            if USE_BF16:
                k_dot = k.to(tl.bfloat16)
            else:
                k_dot = k.to(tl.float16)
            scores = tl.dot(q_dot, tl.trans(k_dot), out_dtype=tl.float32) * sm_scale
            scores = tl.where(valid_n[None, :], scores, -float("inf"))
            scores = tl.where(valid_m[:, None], scores, 0.0)
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(scores - m_ij[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            if USE_BF16:
                pv = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), out_dtype=tl.float32)
            else:
                pv = tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32)
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        out = acc / l_i[:, None]
        tl.store(
            out_ptr
            + b * stride_ob
            + hq * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od,
            out,
            mask=valid_m[:, None] & valid_d[None, :],
        )


    @triton.jit
    def selector_logits_blocked_kernel(
        q_ptr,
        kq_ptr,
        ks_ptr,
        kz_ptr,
        kfp_ptr,
        kr_ptr,
        logits_ptr,
        n_quant,
        n_residual,
        stride_qb,
        stride_qh,
        stride_qr,
        stride_qd,
        stride_kqb,
        stride_kqh,
        stride_kqg,
        stride_kqd,
        stride_kqp,
        stride_ksb,
        stride_ksh,
        stride_ksg,
        stride_ksd,
        stride_kfb,
        stride_kfh,
        stride_kfn,
        stride_kfd,
        stride_krb,
        stride_krh,
        stride_krn,
        stride_krd,
        stride_lb,
        stride_lh,
        stride_lr,
        stride_ln,
        sm_scale,
        H_KV: tl.constexpr,
        R: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_N: tl.constexpr,
        KEY_BITS: tl.constexpr,
        KEY_GROUP: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        tile_r = tl.program_id(1)
        tile_n = tl.program_id(2)
        b = pid_bh // H_KV
        h = pid_bh % H_KV
        offs_r = tile_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        valid_r = offs_r < R
        n_total = n_quant + n_residual
        valid_n = offs_n < n_total
        valid_d = offs_d < HEAD_DIM
        q = tl.load(
            q_ptr
            + b * stride_qb
            + h * stride_qh
            + offs_r[:, None] * stride_qr
            + offs_d[None, :] * stride_qd,
            mask=valid_r[:, None] & valid_d[None, :],
            other=0.0,
        )

        if KEY_BITS == 16:
            k = tl.load(
                kfp_ptr
                + b * stride_kfb
                + h * stride_kfh
                + offs_n[:, None] * stride_kfn
                + offs_d[None, :] * stride_kfd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            )
        else:
            is_quant = offs_n < n_quant
            key_group = offs_n // KEY_GROUP
            token_in_group = offs_n % KEY_GROUP
            values_per_byte: tl.constexpr = 8 // KEY_BITS
            byte_id = token_in_group // values_per_byte
            lane = token_in_group % values_per_byte
            packed = tl.load(
                kq_ptr
                + b * stride_kqb
                + h * stride_kqh
                + key_group[:, None] * stride_kqg
                + offs_d[None, :] * stride_kqd
                + byte_id[:, None] * stride_kqp,
                mask=is_quant[:, None] & valid_d[None, :],
                other=0,
            ).to(tl.int32)
            qk_i = (packed >> (lane[:, None] * KEY_BITS)) & ((1 << KEY_BITS) - 1)
            scale_k = tl.load(
                ks_ptr
                + b * stride_ksb
                + h * stride_ksh
                + key_group[:, None] * stride_ksg
                + offs_d[None, :] * stride_ksd,
                mask=is_quant[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            zero_k = tl.load(
                kz_ptr
                + b * stride_ksb
                + h * stride_ksh
                + key_group[:, None] * stride_ksg
                + offs_d[None, :] * stride_ksd,
                mask=is_quant[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            k_quant = qk_i.to(tl.float32) * scale_k + zero_k
            residual_id = offs_n - n_quant
            k_res = tl.load(
                kr_ptr
                + b * stride_krb
                + h * stride_krh
                + residual_id[:, None] * stride_krn
                + offs_d[None, :] * stride_krd,
                mask=(~is_quant)[:, None] & valid_n[:, None] & valid_d[None, :],
                other=0.0,
            )
            k = tl.where(is_quant[:, None], k_quant, k_res)

        if USE_BF16:
            q_dot = q.to(tl.bfloat16)
            k_dot = k.to(tl.bfloat16)
        else:
            q_dot = q.to(tl.float16)
            k_dot = k.to(tl.float16)
        scores = tl.dot(q_dot, tl.trans(k_dot), out_dtype=tl.float32) * sm_scale
        tl.store(
            logits_ptr
            + b * stride_lb
            + h * stride_lh
            + offs_r[:, None] * stride_lr
            + offs_n[None, :] * stride_ln,
            scores,
            mask=valid_r[:, None] & valid_n[None, :],
        )


    @triton.jit
    def selector_logits_kernel(
        q_ptr,
        kq_ptr,
        ks_ptr,
        kz_ptr,
        kfp_ptr,
        kr_ptr,
        logits_ptr,
        n_quant,
        n_residual,
        stride_qb,
        stride_qh,
        stride_qr,
        stride_qd,
        stride_kqb,
        stride_kqh,
        stride_kqg,
        stride_kqd,
        stride_kqp,
        stride_ksb,
        stride_ksh,
        stride_ksg,
        stride_ksd,
        stride_kfb,
        stride_kfh,
        stride_kfn,
        stride_kfd,
        stride_krb,
        stride_krh,
        stride_krn,
        stride_krd,
        stride_lb,
        stride_lh,
        stride_lr,
        stride_ln,
        sm_scale,
        H_KV: tl.constexpr,
        R: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        KEY_BITS: tl.constexpr,
        KEY_GROUP: tl.constexpr,
    ):
        pid_bhr = tl.program_id(0)
        tile_n = tl.program_id(1)
        b = pid_bhr // (H_KV * R)
        rem = pid_bhr % (H_KV * R)
        h = rem // R
        r = rem % R
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_total = n_quant + n_residual
        valid_n = offs_n < n_total
        q = tl.load(
            q_ptr
            + b * stride_qb
            + h * stride_qh
            + r * stride_qr
            + offs_d * stride_qd,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)

        is_quant = offs_n < n_quant
        if KEY_BITS == 16:
            k = tl.load(
                kfp_ptr
                + b * stride_kfb
                + h * stride_kfh
                + offs_n[:, None] * stride_kfn
                + offs_d[None, :] * stride_kfd,
                mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
        else:
            key_group = offs_n // KEY_GROUP
            token_in_group = offs_n % KEY_GROUP
            values_per_byte: tl.constexpr = 8 // KEY_BITS
            byte_id = token_in_group // values_per_byte
            lane = token_in_group % values_per_byte
            p = tl.load(
                kq_ptr
                + b * stride_kqb
                + h * stride_kqh
                + key_group[:, None] * stride_kqg
                + offs_d[None, :] * stride_kqd
                + byte_id[:, None] * stride_kqp,
                mask=is_quant[:, None] & (offs_d[None, :] < HEAD_DIM),
                other=0,
            ).to(tl.int32)
            qi = (p >> (lane[:, None] * KEY_BITS)) & ((1 << KEY_BITS) - 1)
            sc = tl.load(
                ks_ptr
                + b * stride_ksb
                + h * stride_ksh
                + key_group[:, None] * stride_ksg
                + offs_d[None, :] * stride_ksd,
                mask=is_quant[:, None] & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            ze = tl.load(
                kz_ptr
                + b * stride_ksb
                + h * stride_ksh
                + key_group[:, None] * stride_ksg
                + offs_d[None, :] * stride_ksd,
                mask=is_quant[:, None] & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            k_quant = qi.to(tl.float32) * sc + ze
            residual_id = offs_n - n_quant
            k_res = tl.load(
                kr_ptr
                + b * stride_krb
                + h * stride_krh
                + residual_id[:, None] * stride_krn
                + offs_d[None, :] * stride_krd,
                mask=(~is_quant)[:, None]
                & valid_n[:, None]
                & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            k = tl.where(is_quant[:, None], k_quant, k_res)

        score = tl.sum(k * q[None, :], axis=1) * sm_scale
        tl.store(
            logits_ptr
            + b * stride_lb
            + h * stride_lh
            + r * stride_lr
            + offs_n * stride_ln,
            score,
            mask=valid_n,
        )


    @triton.jit
    def row_logsumexp_kernel(
        logits_ptr,
        lse_ptr,
        n,
        stride_lb,
        stride_lh,
        stride_lr,
        stride_ln,
        stride_eb,
        stride_eh,
        stride_er,
        H_KV: tl.constexpr,
        R: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // (H_KV * R)
        rem = pid % (H_KV * R)
        h = rem // R
        r = rem % R
        m_i = -float("inf")
        l_i = 0.0
        for start_n in tl.range(0, n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            x = tl.load(
                logits_ptr
                + b * stride_lb
                + h * stride_lh
                + r * stride_lr
                + offs_n * stride_ln,
                mask=offs_n < n,
                other=-float("inf"),
            ).to(tl.float32)
            m_ij = tl.maximum(m_i, tl.max(x, axis=0))
            l_i = l_i * tl.exp(m_i - m_ij) + tl.sum(tl.exp(x - m_ij), axis=0)
            m_i = m_ij
        tl.store(lse_ptr + b * stride_eb + h * stride_eh + r * stride_er, m_i + tl.log(l_i))


    @triton.jit
    def reduce_selector_importance_kernel(
        logits_ptr,
        lse_ptr,
        importance_ptr,
        n,
        stride_lb,
        stride_lh,
        stride_lr,
        stride_ln,
        stride_eb,
        stride_eh,
        stride_er,
        stride_ib,
        stride_ih,
        stride_in,
        H_KV: tl.constexpr,
        R: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SOFTMAX: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        tile_n = tl.program_id(1)
        b = pid_bh // H_KV
        h = pid_bh % H_KV
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = offs_n < n
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for r in range(0, R):
            x = tl.load(
                logits_ptr
                + b * stride_lb
                + h * stride_lh
                + r * stride_lr
                + offs_n * stride_ln,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            if SOFTMAX:
                e = tl.load(lse_ptr + b * stride_eb + h * stride_eh + r * stride_er).to(
                    tl.float32
                )
                acc += tl.exp(x - e)
            else:
                acc += x
        acc /= R
        tl.store(
            importance_ptr + b * stride_ib + h * stride_ih + offs_n * stride_in,
            acc,
            mask=valid,
        )


    @triton.jit
    def gather_packed_kv_kernel(
        indices_ptr,
        kq_ptr,
        ks_ptr,
        kz_ptr,
        vq_ptr,
        vs_ptr,
        vz_ptr,
        kfp_ptr,
        vfp_ptr,
        kr_ptr,
        vr_ptr,
        out_k_ptr,
        out_v_ptr,
        n_quant,
        stride_ib,
        stride_ih,
        stride_ik,
        stride_kqb,
        stride_kqh,
        stride_kqg,
        stride_kqd,
        stride_kqp,
        stride_ksb,
        stride_ksh,
        stride_ksg,
        stride_ksd,
        stride_vqb,
        stride_vqh,
        stride_vqn,
        stride_vqg,
        stride_vqp,
        stride_vsb,
        stride_vsh,
        stride_vsn,
        stride_vsg,
        stride_kfb,
        stride_kfh,
        stride_kfn,
        stride_kfd,
        stride_vfb,
        stride_vfh,
        stride_vfn,
        stride_vfd,
        stride_krb,
        stride_krh,
        stride_krn,
        stride_krd,
        stride_vrb,
        stride_vrh,
        stride_vrn,
        stride_vrd,
        stride_okb,
        stride_okh,
        stride_okn,
        stride_okd,
        stride_ovb,
        stride_ovh,
        stride_ovn,
        stride_ovd,
        H_KV: tl.constexpr,
        K: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_BITS: tl.constexpr,
        VALUE_BITS: tl.constexpr,
        KEY_GROUP: tl.constexpr,
        VALUE_GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // (H_KV * K)
        rem = pid % (H_KV * K)
        h = rem // K
        kk = rem % K
        offs_d = tl.arange(0, BLOCK_D)
        idx = tl.load(indices_ptr + b * stride_ib + h * stride_ih + kk * stride_ik).to(tl.int32)
        is_quant = idx < n_quant

        if KEY_BITS == 16:
            kval = tl.load(
                kfp_ptr
                + b * stride_kfb
                + h * stride_kfh
                + idx * stride_kfn
                + offs_d * stride_kfd,
                mask=offs_d < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
        else:
            key_group = idx // KEY_GROUP
            token_in_group = idx % KEY_GROUP
            values_per_byte: tl.constexpr = 8 // KEY_BITS
            byte_id = token_in_group // values_per_byte
            lane = token_in_group % values_per_byte
            p = tl.load(
                kq_ptr
                + b * stride_kqb
                + h * stride_kqh
                + key_group * stride_kqg
                + offs_d * stride_kqd
                + byte_id * stride_kqp,
                mask=is_quant & (offs_d < HEAD_DIM),
                other=0,
            ).to(tl.int32)
            qi = (p >> (lane * KEY_BITS)) & ((1 << KEY_BITS) - 1)
            sc = tl.load(
                ks_ptr
                + b * stride_ksb
                + h * stride_ksh
                + key_group * stride_ksg
                + offs_d * stride_ksd,
                mask=is_quant & (offs_d < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            ze = tl.load(
                kz_ptr
                + b * stride_ksb
                + h * stride_ksh
                + key_group * stride_ksg
                + offs_d * stride_ksd,
                mask=is_quant & (offs_d < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            kqv = qi.to(tl.float32) * sc + ze
            ridx = idx - n_quant
            krv = tl.load(
                kr_ptr
                + b * stride_krb
                + h * stride_krh
                + ridx * stride_krn
                + offs_d * stride_krd,
                mask=(~is_quant) & (offs_d < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            kval = tl.where(is_quant, kqv, krv)

        if VALUE_BITS == 16:
            vval = tl.load(
                vfp_ptr
                + b * stride_vfb
                + h * stride_vfh
                + idx * stride_vfn
                + offs_d * stride_vfd,
                mask=offs_d < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
        else:
            values_per_byte_v: tl.constexpr = 8 // VALUE_BITS
            cg = offs_d // VALUE_GROUP
            cwithin = offs_d % VALUE_GROUP
            byte_id_v = cwithin // values_per_byte_v
            lane_v = cwithin % values_per_byte_v
            pv = tl.load(
                vq_ptr
                + b * stride_vqb
                + h * stride_vqh
                + idx * stride_vqn
                + cg * stride_vqg
                + byte_id_v * stride_vqp,
                mask=is_quant & (offs_d < HEAD_DIM),
                other=0,
            ).to(tl.int32)
            qv = (pv >> (lane_v * VALUE_BITS)) & ((1 << VALUE_BITS) - 1)
            scv = tl.load(
                vs_ptr
                + b * stride_vsb
                + h * stride_vsh
                + idx * stride_vsn
                + cg * stride_vsg,
                mask=is_quant & (offs_d < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            zev = tl.load(
                vz_ptr
                + b * stride_vsb
                + h * stride_vsh
                + idx * stride_vsn
                + cg * stride_vsg,
                mask=is_quant & (offs_d < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            vqv = qv.to(tl.float32) * scv + zev
            ridx_v = idx - n_quant
            vrv = tl.load(
                vr_ptr
                + b * stride_vrb
                + h * stride_vrh
                + ridx_v * stride_vrn
                + offs_d * stride_vrd,
                mask=(~is_quant) & (offs_d < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            vval = tl.where(is_quant, vqv, vrv)

        tl.store(
            out_k_ptr
            + b * stride_okb
            + h * stride_okh
            + kk * stride_okn
            + offs_d * stride_okd,
            kval,
            mask=offs_d < HEAD_DIM,
        )
        tl.store(
            out_v_ptr
            + b * stride_ovb
            + h * stride_ovh
            + kk * stride_ovn
            + offs_d * stride_ovd,
            vval,
            mask=offs_d < HEAD_DIM,
        )


if TRITON_AVAILABLE:

    @triton.jit
    def quantize_key_groups_kernel(
        x_ptr,
        out_q_ptr,
        out_scale_ptr,
        out_zero_ptr,
        token_count,
        stride_xb,
        stride_xh,
        stride_xt,
        stride_xd,
        stride_qb,
        stride_qh,
        stride_qg,
        stride_qd,
        stride_qp,
        stride_sb,
        stride_sh,
        stride_sg,
        stride_sd,
        H_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KEY_BITS: tl.constexpr,
        KEY_GROUP: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_group = tl.program_id(0)
        pid_d = tl.program_id(1)
        groups_per_head = token_count // KEY_GROUP
        b = pid_group // (H_KV * groups_per_head)
        rem = pid_group % (H_KV * groups_per_head)
        h = rem // groups_per_head
        g = rem % groups_per_head
        offs_t = tl.arange(0, KEY_GROUP)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        x = tl.load(
            x_ptr
            + b * stride_xb
            + h * stride_xh
            + (g * KEY_GROUP + offs_t[:, None]) * stride_xt
            + offs_d[None, :] * stride_xd,
            mask=offs_d[None, :] < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        mn = tl.min(x, axis=0)
        mx = tl.max(x, axis=0)
        levels: tl.constexpr = (1 << KEY_BITS) - 1
        scale = tl.maximum((mx - mn) / levels, 1.0e-8)
        tl.store(
            out_scale_ptr
            + b * stride_sb
            + h * stride_sh
            + g * stride_sg
            + offs_d * stride_sd,
            scale,
            mask=offs_d < HEAD_DIM,
        )
        tl.store(
            out_zero_ptr
            + b * stride_sb
            + h * stride_sh
            + g * stride_sg
            + offs_d * stride_sd,
            mn,
            mask=offs_d < HEAD_DIM,
        )
        values_per_byte: tl.constexpr = 8 // KEY_BITS
        packed_t: tl.constexpr = KEY_GROUP // values_per_byte
        offs_p = tl.arange(0, packed_t)
        packed = tl.zeros((packed_t, BLOCK_D), tl.int32)
        for lane in tl.static_range(0, values_per_byte):
            token_lane = offs_p * values_per_byte + lane
            x_lane = tl.load(
                x_ptr
                + b * stride_xb
                + h * stride_xh
                + (g * KEY_GROUP + token_lane[:, None]) * stride_xt
                + offs_d[None, :] * stride_xd,
                mask=offs_d[None, :] < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
            q_lane = tl.maximum(
                0.0,
                tl.minimum(
                    levels,
                    tl.floor((x_lane - mn[None, :]) / scale[None, :] + 0.5),
                ),
            ).to(tl.int32)
            packed += q_lane << (lane * KEY_BITS)
        tl.store(
            out_q_ptr
            + b * stride_qb
            + h * stride_qh
            + g * stride_qg
            + offs_d[None, :] * stride_qd
            + offs_p[:, None] * stride_qp,
            packed,
            mask=offs_d[None, :] < HEAD_DIM,
        )


    @triton.jit
    def quantize_values_kernel(
        x_ptr,
        out_q_ptr,
        out_scale_ptr,
        out_zero_ptr,
        token_count,
        stride_xb,
        stride_xh,
        stride_xt,
        stride_xd,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qg,
        stride_qp,
        stride_sb,
        stride_sh,
        stride_st,
        stride_sg,
        H_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        VALUE_BITS: tl.constexpr,
        VALUE_GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        channel_groups: tl.constexpr = HEAD_DIM // VALUE_GROUP
        b = pid // (H_KV * token_count * channel_groups)
        rem = pid % (H_KV * token_count * channel_groups)
        h = rem // (token_count * channel_groups)
        rem2 = rem % (token_count * channel_groups)
        t = rem2 // channel_groups
        cg = rem2 % channel_groups
        offs_c = tl.arange(0, VALUE_GROUP)
        x = tl.load(
            x_ptr
            + b * stride_xb
            + h * stride_xh
            + t * stride_xt
            + (cg * VALUE_GROUP + offs_c) * stride_xd,
        ).to(tl.float32)
        mn = tl.min(x, axis=0)
        mx = tl.max(x, axis=0)
        levels: tl.constexpr = (1 << VALUE_BITS) - 1
        scale = tl.maximum((mx - mn) / levels, 1.0e-8)
        tl.store(
            out_scale_ptr
            + b * stride_sb
            + h * stride_sh
            + t * stride_st
            + cg * stride_sg,
            scale,
        )
        tl.store(
            out_zero_ptr
            + b * stride_sb
            + h * stride_sh
            + t * stride_st
            + cg * stride_sg,
            mn,
        )
        values_per_byte: tl.constexpr = 8 // VALUE_BITS
        packed_c: tl.constexpr = VALUE_GROUP // values_per_byte
        offs_p = tl.arange(0, packed_c)
        packed = tl.zeros((packed_c,), tl.int32)
        for lane in tl.static_range(0, values_per_byte):
            channel_lane = offs_p * values_per_byte + lane
            x_lane = tl.load(
                x_ptr
                + b * stride_xb
                + h * stride_xh
                + t * stride_xt
                + (cg * VALUE_GROUP + channel_lane) * stride_xd,
            ).to(tl.float32)
            q_lane = tl.maximum(
                0.0,
                tl.minimum(levels, tl.floor((x_lane - mn) / scale + 0.5)),
            ).to(tl.int32)
            packed += q_lane << (lane * VALUE_BITS)
        tl.store(
            out_q_ptr
            + b * stride_qb
            + h * stride_qh
            + t * stride_qt
            + cg * stride_qg
            + offs_p * stride_qp,
            packed,
        )
