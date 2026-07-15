read -p "pswd: " pswd

hostname=4a100_5

source_path=/home/cs21d406/github/offline_mpc/dsrl_model/runs/neg_union_full/safecl/pos_0_neg_10_union_500/osil/norm_lr_0.00001/batch_64_hz_50/alpha_0.01_vbeta_1.0/
target_path=/home/returaj/Documents/research/offline_safe_rl/neg_union_full/pos_0_neg_10_union_500/osil_lfn/

for task in 'AntVelocity' 'PointCircle2' 'AntRun' 'DroneRun' 'SwimmerVelocity' 'AntCircle' 'DroneCircle' 'PointGoal1' 'CarGoal1' 'CarCircle2' ; do
# for task in 'AntCircle' 'DroneCircle' 'CarCircle2'; do
# for task in 'AntVelocity' 'SwimmerVelocity' 'Walker2dVelocity' 'HopperVelocity'; do
    echo "Starting $task"
    for ttype in 'ep_reward' 'ep_cost' 'ep_length' 'ep_worst_cost_0.1' 'ep_worst_cost_0.2' 'ep_worst_cost_0.3' 'ep_worst_cost_0.5' 'dataset'; do
  
       sshpass -p $pswd scp -r $hostname:$source_path/*${task}*/*/seed-00*/*${ttype}_*  $target_path/*${task}*/${ttype}/

       echo "Done ${ttype} for $task" 
    done
done
