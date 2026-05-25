# Metrics summary

- Dataset: `local_csv`
- Model: `hf-internal-testing/tiny-random-distilbert`
- Fine-tuning mode: `full_finetune`
- Seed: `42`
- Device: `cpu`
- Max length: `64`
- Trainable params: `87483` / `87483`

## Validation

- Loss: `0.6926`
- Accuracy: `0.5714`
- Macro-F1: `0.3636`

## Test

- Loss: `0.6932`
- Accuracy: `0.5000`
- Macro-F1: `0.3333`

## Notes

Raw text -> pretrained tokenizer -> pretrained Transformer -> classification head. Best checkpoint selected by validation macro-F1.
