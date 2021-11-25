mv /content/arabic-tts/ espnet/tools
cd espnet/tools/arabic-tts
./install_asc_voice.sh
./tts.sh -i input.txt -o content/output.wav -v asc_festvox
./install_asc_voice.sh
