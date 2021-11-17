model = Wav2Vec2ForCTC.from_pretrained("elgeish/wav2vec2-large-xlsr-53-arabic").eval()
!mkdir /content/mmm
model.save_pretrained("/content/mmm/")
