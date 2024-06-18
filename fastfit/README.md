## Run training:
```bash
python -m fastfit.train\
--per_device_train_batch_size 32\
--per_device_eval_batch_size 8\
--overwrite_output_dir\
--report_to none\
--dataloader_drop_last true\
--evaluation_strategy steps\
--max_text_length 200\ 
--logging_steps 100\
--dataloader_drop_last=False\
--optim adafactor\
--label_column_name label --text_column_name text\
--num_repeats 4\
--clf_loss_factor 0.1 --do_train --fp16\
--num_train_epochs 2\
--train_file ../data/explanation_dataset.json\
--validation_file ../data/explanation_dataset_test.json\
--output_dir ./tmp/paraphrase-mpnet-base-v2\
--model_name_or_path "sentence-transformers/paraphrase-mpnet-base-v2"
```

## Run inference:
```bash
python -m fastfit.infer
```

## Build and run docker image:
```
docker build -f Dockerfile . -t fastfit
```

```
docker run -it --runtime nvidia --gpus \"device=0,1\" --rm fastfit\
--model_name_or_path sentence-transformers/paraphrase-mpnet-base-v2\
--train_file train.json --validation_file val.json\
```
