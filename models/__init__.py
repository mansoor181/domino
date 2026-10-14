"""denoisers and the domino module.

a denoiser maps partly masked rows z [B, L] and diffusion times t [B] (t = 1 is fully masked) to
float32 logits [B, L, V] and final hidden states [B, L, D]. it also exposes `mask_id`, `pad_id`,
`eps` (a position is masked at time t with probability (1 - eps) t), `seq_len`, `vocab_size`,
`dim` and `token_embedding` [V, D].
"""
