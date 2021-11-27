mv /content/arabic-tts/ espnet/tools
bash espnet/tools/arabic-tts/install_asc_voice.sh
bash espnet/tools/arabic-tts/tts.sh -i espnet/tools/arabic-tts/input.txt -o espnet/tools/arabic-tts/output.wav -v asc_festvox
