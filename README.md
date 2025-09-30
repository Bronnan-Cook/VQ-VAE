# Vector Quantized Variational — Automatic Encoders

A complete framework for VQ-VAE-2 by @Google-DeepMind with PixelSnail prior networks, featuring several improvements.

## Features

![Coverage](https://img.shields.io/scrutinizer/coverage/g/Bronnan-Cook/VQ-VAE)  

- [ ] **Hierarchical VQ-VAE**: Multi-scale latent representations for high-resolution sample generation.
- [ ] **Multi-Medium PixelSnail Prior**: Advanced autoregressive model with causal attention.
- [ ] **Class-Conditional Generation**: Support for conditional sample generation.
- [ ] **Rejection Sampling**: Classifier-based quality-diversity trade-off.
- [ ] **Quantitative Metrics**: Provided metrics include FID, Inception Score, MSE, Precision-Recall, CAS, etc.
- [ ] **EMA Codebook Updates**: Stable training with exponential moving averages.
- [ ] **TensorBoard Integration**: Real-time training visualization.
- [ ] **Check Pointing**: Automatic model saving & restoration.
- [ ] **Template Datasets**: Further improved quality & diversity without copyright strikes or medium limitations.

## Usage

```bash
# Access the optional software directory.
cd /opt

# Clone this GitHub Project.
git clone https://github.com/bronnan-cook/vq-vae

# Access the directory of this framework.
cd vq-vae

# Install all requirements.
pip install -r requirements.txt

# Load your configuration files.
python __init__.py path/or/url/to/datasets.yaml

# Supervise the framework.
tensorboard --logdir=/opt/vq-vae/dat/log
```

## Configuration

The following is an example for the FFHQ datasets.

```yaml ffhq_1024.yaml
coders:
  input_sizes: [1024, 1024]
  latent_layers:
    - [32, 32]
    - [64, 64]
    - [128, 128]
  beta: 0.25
  batch_sizes: 128
  hidden_units: 128
  residual_units: 64
  layers: 2
  codebook_sizes: 512
  codebook_dimensions: 64
  encoder_convolution_filter_sizes: 3
  upsampling_convolution_filter_sizes: 4
  training_steps: 304741
prior_networks:
  latent_layer_0:
    batch_sizes: 1024
    hidden_units: 512
    residual_units: 2048
    layers: 20
    attention_layers: 4
    attention_heads: 8
    convolution_filter_sizes: 5
    dropouts: 0.5
    output_stack_layers: 0
    conditioning_stack_residual_blocks: ~
    training_steps: 237000
  latent_layer_1:
    batch_sizes: 512
    hidden_units: 512
    residual_units: 1024
    layers: 20
    attention_layers: 1
    attention_heads: ~
    convolution_filter_sizes: 5
    dropouts: 0.3
    output_stack_layers: 0
    conditioning_stack_residual_blocks: 8
    training_steps: 57400
  latent_layer_2:
    batch_sizes: 256
    hidden_units: 512
    residual_units: 1024
    layers: 10
    attention_layers: 0
    attention_heads: ~
    convolution_filter_sizes: 5
    dropouts: 0.25
    output_stack_layers: 0
    conditioning_stack_residual_blocks: 8
    training_steps: 270000
training:
  ema_decay: 0.99
  use_ema: true
  learning_rate: 3e-4
  save_checkpoint_steps: 10000
  eval_steps: 1000
  datasets: "path/to/datasets"
generating:
  samples: 18
  top_k_percent: 10
  temperature: 0.9
  use_rejection_sampling: true
  classifier_path:
    - ["path/to/classifier", true]
    - ["path/to/another/classifier", false]
metrics:
  use_precision_recall: true
  use_inception_score: true
  use_nll: true
  use_mse: true
  use_cas: true
  use_fid: true
  fid_batch_size: 50
```

## Model Architecture

### VQ-VAE-2 Decoders & Encoders

- Hierarchical downsampling with residual blocks.
- Multi-scale vector quantization.
- Exponential Moving Average codebook updates.

### PixelSnail Prior Networks

- Gated residual blocks with causal convolutions.
- Multi-head self-attention mechanisms.
- Conditional generation support.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{razavi2019vqvae2,
	title={Generating Diverse High-Fidelity Images with VQ-VAE-2},
	author={Razavi, Ali & van den Oord, Aaron and Vinyals, Oriol},
	booktitle={Advances in Neural Information Processing Systems},
	year={2019}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the ![Apache License 2.0](https://img.shields.io/github/license/Bronnan-Cook/VQ-VAE).

## Acknowledgments

### References

- [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://papers.nips.cc/paper_files/paper/2019/hash/5f8e2fa1718d1bbcadf1cd9c7a54fb8c-Abstract.html)

### Contributors

- ![Contributors](https://img.shields.io/github/all-contributors/Bronnan-Cook/VQ-VAE)
- [ZZZ Code AI](https://www.instagram.com/jonathan_magnan/)
- [DeepSeek R1-0528](https://chat.deepseek.com/)

### Dependencies

- ![Dependencies](https://img.shields.io/librariesio/github/Bronnan-Cook/VQ-VAE)
