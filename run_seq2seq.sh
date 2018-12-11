#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=30:00
#PBS -qexpress_gpu


DATE=`date '+%d-%m-%Y_%H-%M-%S'`

source activate torch-py3.6

cp -r $HOME/IR2/UvaIR2/data/ "$TMPDIR"
mkdir "$TMPDIR"/saved_models
mkdir "$TMPDIR"/figures

python seq2seq.py

cp -r "$TMPDIR"/figures $HOME/IR2/UvaIR2/figures
cp -r "$TMPDIR"/saved_models $HOME/IR2/UvaIR2/saved_models
