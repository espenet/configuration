from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
def training():
    training_args = TrainingArguments(
  	group_by_length=True,
  	per_device_train_batch_size=16,
  	gradient_accumulation_steps=2,
  	evaluation_strategy="steps",
  	num_train_epochs=30,
  	fp16=True,
  	save_steps=100,
  	eval_steps=100,
  	logging_steps=10,
  	learning_rate=3e-4,
  	warmup_steps=500,
  	save_total_limit=2,
     )

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)
trainer.train()
