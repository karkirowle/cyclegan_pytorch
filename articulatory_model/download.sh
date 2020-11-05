wget -N  -q https://raw.githubusercontent.com/yhgon/colab_utils/master/gfile.py

python gfile.py -u 'https://drive.google.com/open?id=1tJbBZunkZXxDuU1WmmZqJSW3Yp9teaAM' -f 'mngu0.zip'

python gfile.py -u 'https://drive.google.com/open?id=1lNCFelkEoWf7VdVltqRQPQpv7UVXewH6' -f 'trainfiles.txt'

python gfile.py -u 'https://drive.google.com/open?id=1SuCJ-qDnebJXS9pJXlYRRKtFcj3EAcH3' -f 'validationfiles.txt'

python gfile.py -u 'https://drive.google.com/open?id=1b1Ew4rvqJtJCOOi1e16zUo_6D77TouvX' -f 'testfiles.txt'

unzip mngu0.zip