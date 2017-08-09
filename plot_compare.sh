#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 compare_ind machine_1 time_stamp_1 .. .. machine_N time_stamp_N"
    exit
fi

LOG_FIL_ALL=""
LOG_DIR="logs/"
j=0
for i in $@
do
    if [ $j -eq 0 ]; then
        SUB_DIR="figs/compare_"$i"/"
        let j=$j+1
        continue
    fi
    let j=$j+1
    if [ $(($j % 2)) = 0 ]; then    # machine
        MACHINE=$i
    else                            # time_stamp
        TIME_STAMP=$i
        LOG_FIL=$LOG_DIR$MACHINE"_"$TIME_STAMP".log"
        echo "$LOG_FIL"
        LOG_FIL_ALL="$LOG_FIL_ALL $LOG_FIL"
    fi
done

if [ -d "$SUB_DIR" ]; then
    rm -r $SUB_DIR
fi
mkdir $SUB_DIR

OPT="lmj-plot --input $LOG_FIL_ALL \
              --num-x-ticks 8 \
              --alpha 0.7 \
              --colors r g b m y c \
              --points - \
              -g -T"

$OPT -o $SUB_DIR"p_loss_avg.png"         -m 'Iteration: (\d+); p_loss_avg: (\S+)'         --xlabel Iteration --ylabel p_loss_avg            --title "p_loss_avg"          &
$OPT -o $SUB_DIR"v_loss_avg.png"         -m 'Iteration: (\d+); v_loss_avg: (\S+)'         --xlabel Iteration --ylabel v_loss_avg            --title "v_loss_avg"          &
$OPT -o $SUB_DIR"loss_avg.png"           -m 'Iteration: (\d+); loss_avg: (\S+)'           --xlabel Iteration --ylabel loss_avg              --title "loss_avg"            &
$OPT -o $SUB_DIR"entropy_avg.png"        -m 'Iteration: (\d+); entropy_avg: (\S+)'        --xlabel Iteration --ylabel entropy_avg           --title "entropy_avg"         &
$OPT -o $SUB_DIR"v_avg.png"              -m 'Iteration: (\d+); v_avg: (\S+)'              --xlabel Iteration --ylabel v_avg                 --title "v_avg"               &
$OPT -o $SUB_DIR"reward_avg.png"         -m 'Iteration: (\d+); reward_avg: (\S+)'         --xlabel Iteration --ylabel reward_avg            --title "reward_avg"          &
$OPT -o $SUB_DIR"reward_std.png"         -m 'Iteration: (\d+); reward_std: (\S+)'         --xlabel Iteration --ylabel reward_std            --title "reward_std"          &
$OPT -o $SUB_DIR"steps_avg.png"          -m 'Iteration: (\d+); steps_avg: (\S+)'          --xlabel Iteration --ylabel steps_avg             --title "steps_avg"           &
$OPT -o $SUB_DIR"steps_std.png"          -m 'Iteration: (\d+); steps_std: (\S+)'          --xlabel Iteration --ylabel steps_std             --title "steps_std"           &
$OPT -o $SUB_DIR"nepisodes.png"          -m 'Iteration: (\d+); nepisodes: (\S+)'          --xlabel Iteration --ylabel nepisodes             --title "nepisodes"           &
$OPT -o $SUB_DIR"nepisodes_solved.png"   -m 'Iteration: (\d+); nepisodes_solved: (\S+)'   --xlabel Iteration --ylabel nepisodes_solved      --title "nepisodes_solved"    &
$OPT -o $SUB_DIR"repisodes_solved.png"   -m 'Iteration: (\d+); repisodes_solved: (\S+)'   --xlabel Iteration --ylabel repisodes_solved      --title "repisodes_solved"    &
