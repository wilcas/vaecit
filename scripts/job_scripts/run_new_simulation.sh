set -e -x
METHODS=( 'lfa' 'pca' 'fastica' 'kernelpca' 'mmdvae_batch' 'ae_batch' )
METHOD="${METHODS[$1]}"
LATENT=1
DEPTH=5


python multi_method_sim.py \
  --lv-method=${METHOD} \
  --num-latent=${LATENT} \
  --vae-depth=${DEPTH} \
  --model-dir=""


