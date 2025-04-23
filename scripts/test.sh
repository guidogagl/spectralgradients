dataset=sleepedf

model=tinysleepnet

selected_channels="EEG"
sequence_length=21

num_nodes=1
num_workers=27

batch_size=64

ck="output/model/tiny/sleepedf/fold=-1-epoch=4-step=18814-val_acc=0.84.ckpt"
df=${PHYSIOEXDATA}

test_model  -m $model \
            -d $dataset \
            -sc $selected_channels \
            -df $df \
            -nw $num_workers \
            -bs $batch_size \
            -sl $sequence_length \
            -ck_path $ck
