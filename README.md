# Domino

Code for *Domino: Joint Unmasking via First-order Markov Chain for Few-Step Masked Diffusion*.

A few-step sampler of a masked diffusion model reveals many positions in one step and draws their tokens independently, which ignores the dependence between tokens revealed together. Domino draws the revealed positions of a step as a first-order Markov chain in position order. The first revealed position keeps the posterior of the pre-trained denoiser, and each next position $j$ follows a transition given the token $a$ of its predecessor $i$,

$$p(z^j = b \mid z^i = a) = \mathrm{softmax}_b\left(\ell_j(b) + E_a^\top A_{ij} E_b\right), \qquad A_{ij} = g\left([u_i; u_j; e(j - i)]\right),$$

where $\ell_j$ are the logits of the denoiser, $u = W_u h$ projects its final hidden states, $E$ projects its input embedding to rank $d_p$, $e$ embeds the signed gap between the two positions, and $g$ is a two-layer network. The last layer of $g$ starts at zero, so an untrained module recovers the factorized step of the denoiser. The denoiser stays fixed, and we train only the module, which adds about 4.5M parameters to a denoiser of 170M. We integrate Domino into MDLM, SEDD and SDTT on text and molecules, and into MaskGIT on ImageNet.

## Layout

```
models/     domino.py      the chain: likelihood, training losses, draws and the coupling for maskgit
            dit.py         mdlm's diffusion transformer
            mdlm.py        mdlm on top of it, with its training loss
            sedd.py        a wrapper of the released sedd code
            maskgit.py     a wrapper of the released maskgit code
sampling/   sampler.py     the K-step sampler for text and molecules, with its unmasking strategies
            maskgit.py     maskgit's sampler with domino at the revealed positions
training/   train_mdlm.py, train_sdtt.py, train_domino.py
eval/       evaluate.py (text and molecules), evaluate_images.py, mauve.py, blimp.py
metrics/    text.py, mols.py, images.py
data/       loaders of the prepared corpora
scripts/    setup, data preparation, checkpoint conversion, timing and examples
configs/    mdlm/, sdtt/ and domino/
patches/    two small patches to the mdlm and sedd repositories
tests/      checks of the chain on the cpu
```

## Setup

We ran everything with Python 3.10, PyTorch 2.2 and CUDA 12.1 on one H100 GPU.

```bash
pip install -r requirements.txt
bash scripts/setup_external.sh
```

The setup script clones MDLM, SEDD and MaskGIT-pytorch at the commits we used into `external/` and applies the patches in `patches/`. The MDLM patch loads LM1B from a parquet mirror, since the original loading script no longer runs under `datasets` 4, and the SEDD patch does the same and adds the molecule corpora.

By default, data and outputs live in the repository, under `storage/` and `results/`. To place them elsewhere, set these variables in the environment or in a `.env` file next to `paths.py`:

| variable | default | contents |
|---|---|---|
| `DOMINO_TEXT_ROOT` | `storage/text` | text caches |
| `DOMINO_MOL_ROOT` | `storage/mols` | QM9 and ZINC-250k caches |
| `DOMINO_IMAGE_ROOT` | `storage/images` | ImageNet-256 tokens, MaskGIT and its VQGAN, FID references |
| `DOMINO_EXTERNAL` | `external` | the cloned repositories |
| `DOMINO_RESULTS` | `results` | checkpoints and evaluations |

Config files refer to these roots as `${DOMINO_RESULTS}` and so on. We run all commands from the repository root.

## Data

**Text.** WikiText-103 and LM1B go through MDLM's own data pipeline, so the caches match the ones MDLM trains on.

```bash
python -m scripts.prep_text --corpus wikitext103
python -m scripts.prep_text --corpus lm1b
python -m scripts.prep_text --corpus lm1b_gpt2
python -m scripts.prep_text --corpus openwebtext --snapshot <directory of parquet shards>
```

`lm1b_gpt2` is LM1B in the GPT-2 tokenization of our SEDD model. For OpenWebText, only the module trains on the corpus, so we build a bounded subset of 200k rows of 1,024 tokens from all but the last parquet shard of `Skylion007/openwebtext`, and 2,048 validation rows from the last shard. Download the shards first with `huggingface-cli download Skylion007/openwebtext --repo-type dataset --revision refs/convert/parquet --local-dir <dir>`, and pass the directory that holds the `.parquet` files.

**Molecules.** We use the QM9 and ZINC-250k preprocessing of [PairFlow](https://github.com/KAIST-Visual-AI-Group/PairFlow) (commit `0bdc543`). Run `python data/preprocessed/qm9/download_dataset.py` and `python data/preprocessed/zinc-250k/download_dataset.py` in that repository, and copy `data/preprocessed/qm9` and `data/preprocessed/zinc-250k` with their `parsed/` folders to `$DOMINO_MOL_ROOT`.

**Images.** The first command downloads MaskGIT and its VQGAN from `llvictorll/Maskgit-pytorch` and the ImageNet-256 token dump `Jiayi-Pan/imagenet-maskgit-256-vqgan`, and writes the token caches. The second computes the Inception features of the 50,000 validation images, which serve as the reference of all FID scores, and prints the FID of the tokenizer's own reconstructions.

```bash
python -m scripts.prep_imagenet256
python -m scripts.build_image_refs
```

## Pre-trained denoisers

**MDLM**, trained from scratch for 200k steps on text, 50k on QM9 and 100k on ZINC-250k:

```bash
python -m training.train_mdlm --config configs/mdlm/wikitext103.yaml    # or lm1b, qm9, zinc250k
```

**SEDD** (absorbing graph), trained with SEDD's own trainer:

```bash
cd external/sedd
python train.py ngpus=1 model=small model.length=128 data.train=wikitext103 data.valid=wikitext103 \
    data.cache_dir=$DOMINO_TEXT_ROOT/sedd training.batch_size=512 training.accum=4 \
    training.n_iters=200000 optim.lr=3e-4 hydra.run.dir=$DOMINO_RESULTS/sedd/wikitext103
```

For LM1B, replace `wikitext103` by `lm1b`. For the molecules, export `DOMINO_MOL_ROOT` as an absolute path and run

```bash
python train.py ngpus=1 tokens=40 model.length=32 data.train=qm9 data.valid=qm9 graph.type=absorb \
    noise.type=loglinear model.hidden_size=768 model.n_blocks=12 model.n_heads=12 model.cond_dim=128 \
    model.dropout=0.1 training.batch_size=512 training.accum=1 training.n_iters=50000 \
    training.snapshot_sampling=False eval.perplexity=False eval.batch_size=512 optim.lr=3e-4 \
    optim.warmup=2500 hydra.run.dir=$DOMINO_RESULTS/sedd/qm9
```

with `tokens=72 model.length=74 training.accum=2 training.n_iters=100000` for ZINC-250k.

**SDTT** students, distilled from our MDLM models with SDTT's released recipe:

```bash
python -m training.train_sdtt --config configs/sdtt/wikitext103.yaml    # or lm1b, qm9, zinc250k
```

**Released OpenWebText models.** We convert the released MDLM and SDTT models into our format, and load SEDD from the hub as it is (`louaaron/sedd-small`).

```bash
python -m scripts.convert_mdlm --repo kuleshov-group/mdlm-owt --out $DOMINO_RESULTS/mdlm/openwebtext.pt
python -m scripts.convert_mdlm --repo jdeschena/sdtt --revision baselines_kld_step_70000 \
    --out $DOMINO_RESULTS/sdtt/openwebtext.pt
```

**MaskGIT** comes with `scripts.prep_imagenet256`.

## Training Domino

```bash
python -m training.train_domino --config configs/domino/wikitext103_mdlm.yaml
```

`configs/domino/` holds one config per corpus and denoiser (`{wikitext103,lm1b,openwebtext,qm9,zinc250k}_{mdlm,sedd,sdtt}.yaml`) and the two image modules. The image module trains first on random reveal sets (`imagenet256_random.yaml`) and then continues on the reveal sets of MaskGIT's own sampler (`imagenet256_confidence.yaml`). A config can name another one under `base`, and any setting can be changed from the command line, for example `--override seed=1 out_dir=results/domino/qm9_mdlm_s1`. On WikiText-103, a module trains in about 1.3 H100 hours. The log records the loss of the chain next to the loss of the factorized step on the same revealed positions, and at intervals of 1,000 steps the $K$-step likelihood bound of both kernels on held-out rows.

## Evaluation

**Text.** The evaluator samples three draws of 512 sequences at 1, 2, 3, 4, 8, 16 and 32 steps, and reports the generative perplexity under GPT-2 Large and the per-position entropy of the samples, next to the entropy of held-out text.

```bash
python -m eval.evaluate --denoiser mdlm --ckpt results/mdlm/wikitext103/ckpt_0200000.pt \
    --module results/domino/wikitext103_mdlm/module_0025000.pt \
    --dataset wikitext103 --out results/eval/wikitext103_mdlm_domino.json
```

Without `--module`, the same command evaluates the denoiser with its own factorized step. A SEDD model takes `--denoiser sedd --ckpt <run directory or hub id>`, and an SDTT student takes `--denoiser mdlm --ckpt results/sdtt/<corpus>/student_final.pt`. On OpenWebText, we use `--samples 1024 --batch 32 --draws 1`, and `--temperature 0.9` tempers the logits and the potentials alike. Each draw sets its own random seed, but the random stream also depends on `--batch`, which ranged from 32 to 128 in our runs.

**Unmasking strategies.** `--orders random,dilated,confidence,margin,entropy,drawn,eb<gamma>` replaces the random reveal sets with the dilated scheduler, the confidence, margin and entropy strategies, MaskGIT's strategy, or the EB-Sampler with entropy bound `gamma` (for example `eb1.0`).

**Molecules.** The same evaluator reports validity, uniqueness and novelty per draw, and uniqueness, novelty and the Fréchet ChemNet distance (FCD) on the first 10,000 samples of the draws at the largest number of steps. We trained every molecule model with three seeds and report the mean.

```bash
python -m eval.evaluate --denoiser mdlm --ckpt results/mdlm/qm9/ckpt_0050000.pt \
    --module results/domino/qm9_mdlm/module_0025000.pt --dataset qm9 \
    --nfe 1,2,4,8,16,32 --samples 4096 --batch 512 --out results/eval/qm9_mdlm_domino.json
```

**Images.** FID and Inception Score on 10,000 samples at 2 to 8 steps, with MaskGIT's released sampler (guidance 3, temperature 1) and Domino coupled to its candidates. Without `--module`, the command evaluates MaskGIT alone. The last command reveals random positions with the null class, as in our text experiments.

```bash
python -m eval.evaluate_images --module results/domino/imagenet256_confidence/module_0006000.pt \
    --out results/eval/imagenet256_domino.json
python -m eval.evaluate_images --out results/eval/imagenet256_maskgit.json
python -m eval.evaluate_images --module results/domino/imagenet256_random/module_0012000.pt \
    --reveal random --uncond --cfg_w 0 --sm_temp 2 --out results/eval/imagenet256_random.json
```

**MAUVE.** Save the samples with `--save_texts`, then compare them with held-out text. The script also reports the per-sample token entropy and the share of distinct bigrams, and a last row of held-out text against itself.

```bash
python -m eval.evaluate ... --save_texts --out results/eval/wikitext103_mdlm_domino.json
python -m eval.mauve --dataset wikitext103 --out results/eval/mauve_wikitext103.json \
    --set "MDLM + Domino, 4 steps=results/eval/wikitext103_mdlm_domino/random_nfe4_draw*.json"
```

**Subject-verb agreement.** The benchmark needs the BLiMP data and NLTK's tagger.

```bash
git clone https://github.com/alexwarstadt/blimp external/blimp
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
python -m eval.blimp --ckpt results/mdlm/wikitext103/ckpt_0200000.pt \
    --module results/domino/wikitext103_mdlm/module_0025000.pt --dataset wikitext103 \
    --blimp_dir external/blimp/data --out results/eval/blimp_wikitext103.json
```

**Sampling time and examples.**

```bash
python -m scripts.time_sampler --denoiser mdlm --ckpt <mdlm> --module <module>
python -m scripts.examples_text --ckpt <mdlm> --module <module> --dataset lm1b --K 4 --out examples_lm1b.json
python -m scripts.examples_mols --ckpt <mdlm> --module <module> --dataset qm9 --out examples_qm9
python -m scripts.examples_images --module <module> --out examples_imagenet.png
```

## Tests

```bash
python -m pytest tests
```

The tests check on a small random model that the chain is normalized, that its draws follow its likelihood, that an untrained module recovers the factorized step, and that the coupling with MaskGIT keeps the candidates when the potentials vanish.

## Acknowledgments

This code builds on the following work, and we thank the authors for releasing their code and models.

- [MDLM](https://github.com/kuleshov-group/mdlm) (Sahoo et al., 2024): the diffusion transformer that `models/dit.py` ports, the training objective, the text data pipeline, and the released OpenWebText model. The transformer follows DiT (Peebles and Xie, 2023).
- [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) (Lou et al., 2024): the absorbing SEDD models, their trainer, the released `louaaron/sedd-small`, and the LM1B detokenizer in `scripts/prep_text.py`.
- [SDTT](https://github.com/jdeschena/sdtt) (Deschenaux and Gulcehre, 2025): the distillation objective that `training/train_sdtt.py` transcribes, and the released OpenWebText student.
- [MaskGIT-pytorch](https://github.com/llvictorll/Maskgit-pytorch) (Besnier and Chen, 2023), a reproduction of MaskGIT (Chang et al., 2022): the ImageNet transformer, the VQGAN, the sampler that `sampling/maskgit.py` follows, and the Inception metrics. The ImageNet-256 tokens come from `Jiayi-Pan/imagenet-maskgit-256-vqgan`.
- [PairFlow](https://github.com/KAIST-Visual-AI-Group/PairFlow) (Park et al., 2025): the QM9 and ZINC-250k preprocessing and the molecule evaluation conventions.
- The unmasking strategies follow Kim et al. (2025) for confidence, margin and entropy, the dilated unmasking scheduler of Luxembourg et al. (2025), and the EB-Sampler of Ben-Hamu et al. (2025). The coupling for MaskGIT is the maximal coupling of speculative decoding (Leviathan et al., 2023).
- MAUVE (Pillutla et al., 2021), BLiMP (Warstadt et al., 2020), and the FCD (Preuer et al., 2018) through `fcd_torch`.
