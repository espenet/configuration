model = Wav2Vec2ForCTC.from_pretrained("elgeish/wav2vec2-large-xlsr-53-arabic").eval()
model.save_pretrained("/content/espnet/tools/model")
