conda activate melanoma_nn \
  && export CUDA_VISIBLE_DEVICES=MIG-66b51d2d-2e55-5ef5-9a4a-5e318fab19eb \
  && jupyter nbconvert --execute --to notebook ResNet18Regularization_V0.ipynb
