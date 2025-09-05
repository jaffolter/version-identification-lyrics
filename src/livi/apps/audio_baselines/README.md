
# CLEWS Audio Baselines - Inference & Training

This code is based on the official CLEWS repository and paper:

- GitHub: [github.com/serratus-music/clews](https://github.com/serratus-music/clews)
- Paper: [Supervised Contrastive Learning from Weakly-Labeled Audio Segments for Musical Version Matching (Serrà et al., 2025)](https://arxiv.org/abs/2502.16936)

Authors: Joan Serrà, R. Oguz Araz, Dmitry Bogdanov, & Yuki Mitsufuji

---

## How to use (inference)

You can compute audio embeddings from a folder of audio files using a pre-trained checkpoint. Model checkpoints can be downloaded from [Zenodo](https://zenodo.org/records/15045900) and should be placed in the `checkpoints/` folder at the root of the project.

Run the following command in your terminal (Python 3.10+ recommended):

```bash
poetry run python -m livi.apps.audio_baselines.cli infer --dataset covers80 --model-name dvinet
```

This will recursively process all audio files in `<AUDIO_DIR>/<dataset>/` and save the embeddings in `<EMBEDDINGS_DIR>/audio_baselines/<model_name>/<dataset>/`.

You can override the input/output/config/checkpoint paths with the corresponding arguments:

```bash
poetry run python -m livi.apps.audio_baselines.cli infer --dataset covers80 --model-name dvinet --path-in /path/to/audio --path-out /path/to/output --config /path/to/config.yaml --checkpoint /path/to/model.ckpt
```

All options are documented in the CLI (`--help`).

---

## Reference

If you use this code, please cite:

J. Serrà, R. O. Araz, D. Bogdanov, & Y. Mitsufuji (2025). Supervised Contrastive Learning from Weakly-Labeled Audio Segments for Musical Version Matching. ArXiv: 2502.16936.

[[arxiv](https://arxiv.org/abs/2502.16936)] [[github](https://github.com/serratus-music/clews)]
