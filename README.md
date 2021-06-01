# Sparse Gaussian Processes Revisited: Bayesian Approaches to Inducing-Variable Approximations



```bash
>> python3 run_regression.py --help
usage: run_regression.py [-h] [--num_inducing NUM_INDUCING]
                         [--minibatch_size MINIBATCH_SIZE]
                         [--iterations ITERATIONS] [--n_layers N_LAYERS]
                         --dataset DATASET [--fold FOLD]
                         [--prior_type {determinantal,normal,strauss,uniform}]
                         [--model {bsgp}]
                         [--num_posterior_samples NUM_POSTERIOR_SAMPLES]
                         [--step_size STEP_SIZE]

Run regression experiment

optional arguments:
  -h, --help            show this help message and exit
  --num_inducing NUM_INDUCING
  --minibatch_size MINIBATCH_SIZE
  --iterations ITERATIONS
  --n_layers N_LAYERS
  --dataset DATASET
  --fold FOLD
  --prior_type {determinantal,normal,strauss,uniform}
  --model {bsgp}
  --num_posterior_samples NUM_POSTERIOR_SAMPLES
  --step_size STEP_SIZE
```


To reproduce figures, plots and results make sure you have the correct requirements.


## Ablation on priors for BSGP

![](/results/figures/ablation-bsgp-priors.png)


## Ablation on inference variables

![](/results/figures/ablation-inference.png)


## Comparison of objectives

![](/results/figures/comparison-objectives.png)


## Training time

![](/results/figures/train-time.png)



### Reference
Rossi, S., Heinonen, M., Bonilla, E., Shen, Z. &amp; Filippone, M.. (2021).  Sparse Gaussian Processes Revisited: Bayesian Approaches to Inducing-Variable Approximations. <i>Proceedings of The 24th International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 130:1837-1845 