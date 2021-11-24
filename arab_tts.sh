!mv /content/arabic-tts/ espnet/tools
%cd espnet/tools/content/arabic-tts
!./install_asc_voice.sh
!./tts.sh -i input.txt -o output.wav -v asc_festvox
!./install_asc_voice.sh
